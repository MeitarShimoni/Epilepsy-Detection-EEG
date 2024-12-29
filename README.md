# EEG Seizure Detection Using DWT and SVM

This repository implements a real-time epilepsy seizure detection system using EEG data. The system utilizes Discrete Wavelet Transform (DWT) for feature extraction and Support Vector Machine (SVM) for classification. It is based on the methodology described in a 2022 paper on real-time seizure detection.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Datasets](#datasets)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Visualization](#visualization)
- [References](#references)

---

## Overview
This project aims to detect seizures using EEG signals. The primary steps involved are:
1. Preprocessing EEG signals with DWT.
2. Extracting features from each DWT sub-band.
3. Training an SVM model for classification into three classes:
   - Healthy control
   - Seizure-free
   - Seizure-active

---

## Features
- **Preprocessing**: Removes noise and decomposes signals using DWT.
- **Feature Extraction**: Extracts time-domain and frequency-domain features.
- **Classification**: Trains an SVM classifier for high-accuracy seizure detection.
- **Visualization**: Displays performance metrics, including a confusion matrix.

---

## Datasets
The project uses two datasets:
- **Dataset UB**: Short-term EEG recordings from the University of Bonn.
- **Dataset CHB-MIT**: Long-term EEG recordings from Boston Children’s Hospital (optional).

> Note: Ensure you have permission to use the datasets. Follow ethical guidelines when handling medical data.

---

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/EEG-Seizure-Detection-DWT-SVM.git
   ```
2. Open MATLAB and navigate to the cloned directory.
3. Install any required MATLAB toolboxes (e.g., Signal Processing Toolbox).

---

## Usage
### Preprocessing and Feature Extraction
Run the following script to preprocess the EEG signals and extract features:
```matlab
run('feature_extraction.m')
```
### Train and Test the Model
Run the SVM model training and testing script:
```matlab
run('svm_model_training.m')
```
### Visualization
Add your confusion matrix plot code to visualize classification performance:
```matlab
confusionchart(y_test, y_pred);
```

---

## Results
| Metric        | Value   |
|---------------|---------|
| Accuracy      | 97%     |
| Sensitivity   | 96.67%  |

The system achieved 97% accuracy and 96.67% sensitivity on the UB dataset.

### Confusion Matrix
Include your confusion matrix graph here:

![Confusion Matrix Placeholder](path/to/confusion_matrix.png)

---

## Visualization
To visualize the performance:
1. Add the confusion matrix code to the script.
2. Save the confusion matrix plot to the repository.

---

## References
- M. Shen et al., "An EEG based real-time epilepsy seizure detection approach using discrete wavelet transform and machine learning methods," *Biomedical Signal Processing and Control,* vol. 77, 2022.

---

Feel free to contribute by creating pull requests or reporting issues!
