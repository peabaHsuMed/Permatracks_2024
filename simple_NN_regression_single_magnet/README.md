# Prediction the position of single magnet with Simple NN Regression Model

This folder contains the implementation of using a simple NN regression model for predicting single magnet position. The model is trained using data collected on the old grid plate (spacing: 30mm). The training data includes measurements taken at heights ranging from **39.31mm to 79.31mm** over the grid, with each position having 5 different orientations, resulting in a total of **14,580 samples**.

## Important Notes
- The model is **only accurate** when used with the old device (spacing: 30mm), as the training data was collected using this device.
- For accurate results, ensure the data collection and predictions are performed on the same type of grid plate.

## Training Data Details
- Grid plate spacing: **30mm**
- Height range: **39.31mm to 79.31mm**
- Orientations per position: **5**
- Total samples: **14,580**

## How to Execute the Code

1. Navigate to the project directory:
   $ cd OneDrive/desktop/Peaba workspace/Permatracks/sinple_NN_regression_single_magnet
2. Run the localization script:
   $ python localizatio_NN_regression.py

## Raw Data Collection Instructions
1. Press the spacebar to collect raw data.
2. After pressing space 5 times, the collected data will be saved into a file named data_acquisition.csv

