# Prediction of Single Magnet Position with ResNet20 Model

This folder contains the implementation of a **ResNet20 model** for predicting the position of a single magnet. The model is trained using data collected on the old grid plate (spacing: 30mm). Training data includes measurements taken at heights ranging from **39.31mm to 79.31mm** over the grid, with each position having **5 different orientations**, resulting in a total of **14,580 samples**.

## Important Notes
- The model is **only accurate** when used with the old device (spacing: 30mm) since the training data was collected using this setup.
- The ResNet20 model is **slightly more accurate** than the simple NN regression model.

## Training Data Details
- **Grid Plate Spacing**: 30mm
- **Height Range**: 39.31mm to 79.31mm
- **Orientations per Position**: 5
- **Total Samples**: 14,580

## How to Execute the Code

1. Navigate to the project directory:
   $ cd OneDrive/desktop/Peaba workspace/Permatracks/ResNet20_single_magnet
2. Run the localization script:
   $ python localizatio_ResNet20_single_mag.py
