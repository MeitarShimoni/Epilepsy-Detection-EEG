% ClassTestSVM: Three-Class Classification Using DWT Coefficients
disp('--- Starting SVM Classification Process ---');

% Step 1: Dataset Preparation
disp('Step 1: Preparing Dataset...');
folders = {'C:\Users\Meitar Shimoni\MATLAB Drive\EEG_Project\Data\Z', ...
           'C:\Users\Meitar Shimoni\MATLAB Drive\EEG_Project\Data\O', ...
           'C:\Users\Meitar Shimoni\MATLAB Drive\EEG_Project\Data\N', ...
           'C:\Users\Meitar Shimoni\MATLAB Drive\EEG_Project\Data\F', ...
           'C:\Users\Meitar Shimoni\MATLAB Drive\EEG_Project\Data\S'};
labels = [1, 1, 2, 2, 3]; % Class labels: Healthy, Seizure-Free, Seizure-Active

% Sampling frequency for Dataset UB
fs = 173.61;

% Initialize feature matrix and label vector
featureMatrix = [];
labelVector = [];

% Loop through each folder and process EEG segments
for i = 1:length(folders)
    disp(['Processing folder: ', folders{i}]);
    files = dir(fullfile(folders{i}, '*.txt')); % Get all files in folder
    if isempty(files)
        fprintf('No files found in folder: %s\n', folders{i});
    end
    for j = 1:length(files)
        % Load EEG signal
        filePath = fullfile(files(j).folder, files(j).name);
        signal = load(filePath);
        disp(['  Processing file: ', files(j).name, ' (Signal Length: ', num2str(length(signal)), ')']);

        % Perform DWT on the signal
        disp('    Performing DWT...');
        level = 5;
        wavelet = 'db4';
        [CA, CD] = wavedec(signal, level, wavelet);

        % Extract coefficients for sub-bands
        D1 = wrcoef('d', CA, CD, wavelet, 1);
        D2 = wrcoef('d', CA, CD, wavelet, 2);
        D3 = wrcoef('d', CA, CD, wavelet, 3);
        D4 = wrcoef('d', CA, CD, wavelet, 4);
        A4 = wrcoef('a', CA, CD, wavelet, 4);

        % Extract features for each sub-band
        disp('    Extracting features from sub-bands...');
        features_D1 = extractEEGFeatures(D1, fs);
        features_D2 = extractEEGFeatures(D2, fs);
        features_D3 = extractEEGFeatures(D3, fs);
        features_D4 = extractEEGFeatures(D4, fs);
        features_A4 = extractEEGFeatures(A4, fs);

        % Combine features from all sub-bands
        featureVector = [features_D1, features_D2, features_D3, features_D4, features_A4];
        disp(['    Feature vector length: ', num2str(length(featureVector))]);

        % Append features and label
        featureMatrix = [featureMatrix; featureVector];
        labelVector = [labelVector; labels(i)];
    end
end

% Convert labels to categorical for classification
disp('Step 1 Complete: Dataset Prepared.');
labelVector = categorical(labelVector);

% Display Class Distribution
disp('Class Distribution:');
disp(countcats(labelVector));

% Step 2: Split Data into Training (80%) and Testing (20%) Sets
disp('Step 2: Splitting Dataset into Training and Testing...');
cv = cvpartition(size(featureMatrix, 1), 'HoldOut', 0.2);
X_train = featureMatrix(training(cv), :);
Y_train = labelVector(training(cv));
X_test = featureMatrix(test(cv), :);
Y_test = labelVector(test(cv));
disp('Step 2 Complete: Dataset Split.');

% Step 3: Train Classifier (SVM)
disp('Step 3: Training SVM Classifier...');
svmTemplate = templateSVM('KernelFunction', 'rbf', 'BoxConstraint', 1, 'KernelScale', 'auto');
svmModel = fitcecoc(X_train, Y_train, 'Learners', svmTemplate);
disp('Step 3 Complete: SVM Trained.');
% Step 4: Evaluate Model
disp('Step 4: Evaluating Model...');
Y_pred = predict(svmModel, X_test);
confMat = confusionmat(Y_test, Y_pred);
accuracy = sum(diag(confMat)) / sum(confMat, 'all');

% Display Results
disp('Confusion Matrix:');
disp(confMat);
fprintf('Accuracy: %.2f%%\n', accuracy * 100);

% Visualize Confusion Matrix
disp('Visualizing Confusion Matrix...');
confusionchart(confMat, {'Class 1', 'Class 2', 'Class 3'});

disp('--- SVM Classification Process Complete ---');
