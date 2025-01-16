# EEG Seizure Detection with Machine Learning

## Overview

This project focuses on detecting seizures from EEG signals using advanced signal processing techniques and machine learning. The goal is to automate seizure identification, enhancing diagnostic efficiency.

## Methodology

- **Signal Processing**: Leveraged Discrete Wavelet Transform (DWT) to extract spectral features from EEG signals, highlighting critical frequency bands related to seizure activity.
- **Feature Engineering**: Developed robust features for classifying EEG segments into seizure or non-seizure states.
- **Machine Learning**: Implemented a RUSBoost classifier to handle class imbalance and achieve reliable detection.
- **Visualization**: Presented detection results through boolean overlays on EEG signals and band power graphs, showcasing the model's ability to identify seizure activity.

## Visualizations

### EEG Signal with Seizure Detection

The graph below illustrates the raw EEG signal overlaid with a boolean signal (high when seizures are detected). This boolean signal is generated by the trained machine learning model, highlighting periods of seizure activity.

![EEG Signal with Seizure Detection](images/1.png.png)

### Band Power with Seizure Detection

The band power graph below shows the extracted frequency band powers over time, overlaid with a boolean signal generated by the model. This visualization provides insight into the spectral features used for seizure classification.

![Band Power with Seizure Detection](images/2.png.png)


