% Parameters
fs = 256; % Sampling frequency (Hz)
windowSize = 30 * fs; % 30-second window
overlap = 1 * fs; % 1-second overlap
step = windowSize - overlap; % Step size

%"C:\Users\Meitar Shimoni\Desktop\EEG_CHB\rusboostModel.mat"

% Load pre-trained RUSBoost model
modelPath = 'C:\Users\Meitar Shimoni\MATLAB Drive\EEG_Detection\Experiment2\rusboostModel_7_paitentsTrained.mat';
load(modelPath, 'rusboostModel'); % Load the model from the specified path

% File paths
edfFile = 'C:\Users\Meitar Shimoni\MATLAB Drive\EEG_Detection\Experiment2\CHB-MIT\chb01\chb01_04.edf'; % Path to the EDF file

% Load EDF signal
data = edfread(edfFile);
data.Properties.VariableNames = matlab.lang.makeUniqueStrings(data.Properties.VariableNames);

% Select a channel
disp('Available Channels:');
disp(data.Properties.VariableNames);
channelName = data.Properties.VariableNames{1}; % Default to the first available channel
disp(['Using channel: ', channelName]);

% Extract and normalize the signal
signal = cell2mat(data{:, channelName});
if isempty(signal) || all(signal == 0)
    error('Invalid signal (empty or all zeros) for file: %s', edfFile);
end
if std(signal) > 0
    signal = (signal - mean(signal)) / std(signal);
else
    error('Signal in file %s has zero standard deviation.', edfFile);
end

% Initialize boolean signal
booleanSignal = zeros(1, length(signal));

% Generate sliding windows
numWindows = floor((length(signal) - windowSize) / step) + 1;
predictions = zeros(1, numWindows); % Initialize predictions array

disp('Processing windows and predicting seizures...');
for i = 1:numWindows
    % Extract window
    startIdx = (i - 1) * step + 1;
    endIdx = startIdx + windowSize - 1;
    if endIdx > length(signal)
        break;
    end
    segment = signal(startIdx:endIdx);

    % Perform DWT for feature extraction
    [CA, CD] = wavedec(segment, 6, 'db16'); % 6-level DWT
    D4 = wrcoef('d', CA, CD, 'db16', 4); % 8–16 Hz
    D5 = wrcoef('d', CA, CD, 'db16', 5); % 4–8 Hz

    % Extract features
    features_D4 = extractEEGFeatures(D4, fs);
    features_D5 = extractEEGFeatures(D5, fs);
    featureVector = [features_D5, features_D4]; % Combine features

    % Predict seizure using the trained model
    prediction = predict(rusboostModel, featureVector);
    predictions(i) = prediction;

    % Map prediction to boolean signal
    if prediction == 1
        booleanSignal(startIdx:endIdx) = 1;
        
    end
end

% Plot the results
figure;
subplot(2, 1, 1);
plot((1:length(signal)) / fs, signal);
xlabel('Time (s)');
ylabel('Normalized Signal');
title(['Signal from channel: ', channelName]);
tightBounds = 1.2 * max(abs(signal));
ylim([-tightBounds, tightBounds]);
hold on 
timeline = (1:length(booleanSignal)) / fs;
% subplot(2, 1, 2);
plot(timeline, booleanSignal, 'r', 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('Boolean Seizure Signal');
title('Seizure Indication Based on Model');
legend('EEG Signal', 'Seizure Indication');
%ylim([-0.2, 1.2]);

% Plot band power and boolean signal

bandPower = movmean(signal.^2, fs * 2); % Approximate band power with a 2-second moving window
subplot(2, 1, 2);
plot((1:length(bandPower)) / fs, bandPower, 'b');
hold on;
plot(timeline, booleanSignal * max(bandPower), 'r', 'LineWidth', 1.5);
hold off;
xlabel('Time (s)');
ylabel('Band Power and Seizure Indication');
title('Band Power and Boolean Seizure Indication');
legend('Band Power', 'Seizure Indication');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Path to seizure annotations file
seizureFile = strrep(edfFile, '.edf', '.seizures');
seizureTimes = readSeizureFile(seizureFile); % Function to read seizure file

% Initialize ground truth boolean signal
groundTruth = zeros(1, length(signal));

% Map seizure times to the ground truth boolean signal
for i = 1:size(seizureTimes, 1)
    startIdx = round(seizureTimes(i, 1) * fs);
    endIdx = round(seizureTimes(i, 2) * fs);
    groundTruth(startIdx:endIdx) = 1;
end

% Calculate True Positives (TP), False Positives (FP), and False Negatives (FN)
TP = sum((booleanSignal == 1) & (groundTruth == 1));
FP = sum((booleanSignal == 1) & (groundTruth == 0));
FN = sum((booleanSignal == 0) & (groundTruth == 1));
TN = sum((booleanSignal == 0) & (groundTruth == 0));

% Calculate accuracy and sensitivity
accuracy = (TP + TN) / length(signal);
sensitivity = TP / (TP + FN);

fprintf('Accuracy: %.2f%%\n', accuracy * 100);
fprintf('Sensitivity: %.2f%%\n', sensitivity * 100);

% Calculate seizure durations from boolean signal
[~, seizureStartIndices] = find(diff([0 booleanSignal]) == 1);
[~, seizureEndIndices] = find(diff([booleanSignal 0]) == -1);

predictedDurations = (seizureEndIndices - seizureStartIndices) / fs;
fprintf('Predicted Seizure Durations (s):\n');
disp(predictedDurations);

% Calculate seizure durations from ground truth
[~, trueStartIndices] = find(diff([0 groundTruth]) == 1);
[~, trueEndIndices] = find(diff([groundTruth 0]) == -1);

trueDurations = (trueEndIndices - trueStartIndices) / fs;
fprintf('Ground Truth Seizure Durations (s):\n');
disp(trueDurations);

% Helper Function to Read Seizure File
function seizureTimes = readSeizureFile(seizuresFile)
    seizureTimes = [];
    if ~isfile(seizuresFile)
        warning('Seizure file not found: %s', seizuresFile);
        return;
    end
    fid = fopen(seizuresFile, 'r');
    try
        numSeizures = str2double(fgetl(fid)); % First line: number of seizures
        seizureTimes = zeros(numSeizures, 2);
        for i = 1:numSeizures
            times = sscanf(fgetl(fid), '%f %f'); % Start and end times
            seizureTimes(i, :) = times';
        end
    catch
        warning('Error reading seizures file: %s', seizuresFile);
    end
    fclose(fid);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Perform DWT on the entire signal
[CA2, CD2] = wavedec(signal, 6, 'db16'); % 6-level DWT
D4_2 = wrcoef('d', CA2, CD2, 'db16', 4); % 8–16 Hz
D5_2 = wrcoef('d', CA2, CD2, 'db16', 5); % 4–8 Hz

% Parameters for time vector
timeVector = (1:length(signal)) / fs;

% Calculate features for D4
SD_D4 = movstd(D4_2, fs, 'Endpoints', 'fill'); % Moving standard deviation
SE_D4 = movmean(-D4_2.^2 .* log(D4_2.^2 + eps), fs, 'Endpoints', 'fill'); % Moving spectral entropy
LE_D4 = movmean(log(D4_2.^2 + eps), fs, 'Endpoints', 'fill'); % Moving log energy

% Calculate features for D5
SD_D5 = movstd(D5_2, fs, 'Endpoints', 'fill'); % Moving standard deviation
SE_D5 = movmean(-D5_2.^2 .* log(D5_2.^2 + eps), fs, 'Endpoints', 'fill'); % Moving spectral entropy
LE_D5 = movmean(log(D5_2.^2 + eps), fs, 'Endpoints', 'fill'); % Moving log energy

% Plot D4 features
figure;

% SD for D4
subplot(3, 2, 1);
plot(timeVector, SD_D4, 'r');
hold on;
plot(timeVector, booleanSignal * max(SD_D4), 'k', 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('SD');
title('D4: Standard Deviation Over Time');
grid on;

% SE for D4
subplot(3, 2, 3);
plot(timeVector, SE_D4, 'g');
hold on;
plot(timeVector, booleanSignal * max(SE_D4), 'k', 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('SE');
title('D4: Spectral Entropy Over Time');
grid on;

% LE for D4
subplot(3, 2, 5);
plot(timeVector, LE_D4, 'b');
hold on;
plot(timeVector, booleanSignal * max(LE_D4), 'k', 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('LE');
title('D4: Log Energy Over Time');
grid on;

% Plot D5 features
% SD for D5
subplot(3, 2, 2);
plot(timeVector, SD_D5, 'r');
hold on;
plot(timeVector, booleanSignal * max(SD_D5), 'k', 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('SD');
title('D5: Standard Deviation Over Time');
grid on;

% SE for D5
subplot(3, 2, 4);
plot(timeVector, SE_D5, 'g');
hold on;
plot(timeVector, booleanSignal * max(SE_D5), 'k', 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('SE');
title('D5: Spectral Entropy Over Time');
grid on;

% LE for D5
subplot(3, 2, 6);
plot(timeVector, LE_D5, 'b');
hold on;
plot(timeVector, booleanSignal * max(LE_D5), 'k', 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('LE');
title('D5: Log Energy Over Time');
grid on;
