% Parameters
fs = 256; % Sampling frequency (Hz)
windowSize = 30 * fs; % 30-second window
overlap = 1 * fs; % 1-second overlap
step = windowSize - overlap; % Step size

% Directories
dataDir = 'C:\Users\Meitar Shimoni\Desktop\EEG_CHB\CHB-MIT'; % Path to CHB-MIT dataset
patients = {'chb01','chb02','chb03','chb04','chb05','chb06'}; % List of patients

% Initialize storage for features and labels
allFeatures = [];
allLabels = [];

% Function to read seizure intervals from a file
function seizureIntervals = readSeizureFile(seizureFile)
    % Reads the seizure file and extracts seizure intervals
    seizureIntervals = [];
    if exist(seizureFile, 'file') == 2
        % Open the seizure file
        fid = fopen(seizureFile, 'r');
        % Read the first line to get the number of seizures
        numSeizures = fscanf(fid, '%d', 1);
        % Read the subsequent lines for seizure start and end times
        seizureData = fscanf(fid, '%f %f', [2, numSeizures])';
        fclose(fid);
        % Assign seizure intervals
        seizureIntervals = seizureData; % Each row: [start_time, end_time]
    end
end

% Process each patient
for p = 1:length(patients)
    patient = patients{p};

    % List all EDF files for this patient
    edfFiles = dir(fullfile(dataDir, patient, '*.edf'));

    % Process each EDF file
    for f = 1:length(edfFiles)
        edfFile = fullfile(edfFiles(f).folder, edfFiles(f).name);
        seizureFile = strcat(edfFile, '.seizures');

        % Parse seizure intervals for the current file
        seizureIntervals = readSeizureFile(seizureFile);
        if isempty(seizureIntervals)
            warning('No seizure annotations found or parsed for file: %s', edfFile);
            continue;
        end
        disp('Seizure Intervals:');
        disp(seizureIntervals);

        % Load EDF file
        try
            data = edfread(edfFile);
        catch
            warning('Error reading EDF file: %s', edfFile);
            continue;
        end

        % Handle unique signal names
        data.Properties.VariableNames = matlab.lang.makeUniqueStrings(data.Properties.VariableNames);

        % Dynamically select an available channel
        disp('Available Channels:');
        disp(data.Properties.VariableNames);
        channelName = data.Properties.VariableNames{1}; % Default to the first available channel
        disp(['Using channel: ', channelName]);

        % Extract and normalize the signal
        signal = cell2mat(data{:, channelName});
        if isempty(signal) || all(signal == 0)
            warning('Invalid signal (empty or all zeros) for file: %s', edfFile);
            continue;
        end
        disp(['Signal for channel ', channelName, ' loaded with size: ', num2str(size(signal))]);

        if std(signal) > 0
            signal = (signal - mean(signal)) / std(signal);
        else
            warning('Signal in file %s has zero standard deviation.', edfFile);
            continue;
        end

        % Generate sliding windows
        numWindows = floor((length(signal) - windowSize) / step) + 1;
        for i = 1:numWindows
            % Extract window
            startIdx = (i - 1) * step + 1;
            endIdx = startIdx + windowSize - 1;
            segment = signal(startIdx:endIdx);

            % Convert indices to time
            windowStartTime = (startIdx - 1) / fs;
            windowEndTime = (endIdx - 1) / fs;

            % Check if window overlaps with any seizure interval
            isSeizure = any((windowStartTime < seizureIntervals(:, 2)) & ...
                            (windowEndTime > seizureIntervals(:, 1)));

            % Debugging: Print window and seizure information
            disp(['Window Start Time: ', num2str(windowStartTime)]);
            disp(['Window End Time: ', num2str(windowEndTime)]);
            disp(['Is Seizure: ', num2str(isSeizure)]);

            % Perform DWT
            [CA, CD] = wavedec(segment, 6, 'db16'); % 6-level DWT
            D4 = wrcoef('d', CA, CD, 'db16', 4); % 8–16 Hz
            D5 = wrcoef('d', CA, CD, 'db16', 5); % 4–8 Hz

            % Extract features
            features_D4 = extractEEGFeatures(D4, fs);
            features_D5 = extractEEGFeatures(D5, fs);
            featureVector = [features_D5, features_D4]; % Combine features

            % Debugging: Print extracted features
            disp('Extracted Feature Vector:');
            disp(featureVector);

            % Store features and label
            allFeatures = [allFeatures; featureVector];
            allLabels = [allLabels; isSeizure];
        end
    end
end

% Check if any features were extracted
if isempty(allFeatures)
    error('No features were extracted. Ensure seizure annotations are available and correctly parsed.');
end

% Split data into training and testing sets
cv = cvpartition(size(allFeatures, 1), 'HoldOut', 0.2); % 20% for testing
trainFeatures = allFeatures(training(cv), :);
trainLabels = allLabels(training(cv));
testFeatures = allFeatures(test(cv), :);
testLabels = allLabels(test(cv));

% Train RUSBoost Model
disp('Training RUSBoost Model...');
rusboostModel = fitensemble(trainFeatures, trainLabels, 'RUSBoost', 100, 'Tree', ...
    'Type', 'Classification', 'Learners', templateTree('MaxNumSplits', 20));
disp('RUSBoost Model Training Complete.');

% Save the trained model
save('rusboostModel_7_paitentsTrained.mat', 'rusboostModel');

% Evaluate the model
disp('Evaluating RUSBoost Model...');
predictedLabels = predict(rusboostModel, testFeatures);
accuracy = sum(predictedLabels == testLabels) / length(testLabels) * 100;
fprintf('Test Accuracy: %.2f%%\n', accuracy);

% Display confusion matrix
confMat = confusionmat(testLabels, predictedLabels);
disp('Confusion Matrix:');
disp(confMat);

% Visualize confusion matrix
confusionchart(testLabels, predictedLabels);
