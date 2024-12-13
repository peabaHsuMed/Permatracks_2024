# Prediction the position of 2 magnets with Simple NN Regression Model

This folder contains the implementation of using simple NN regression model(s) for predicting the positions of **two magnets**. The models are trained using data collected on the old grid plate (spacing: 30mm).

## Models Available for Testing

### 1. `z1_44_z2_44_regression_model_lr0.001_epochs200_batch16_folds15.h5`
- **Training Data Collection**: 
  - Both magnets are placed at the same height (**44.27mm**).
  - Only **one orientation** per magnet is used for training.
- **Scaler**: `scaler_regression_z1_44_z2_44_1ori.joblib`

### 2. `z1_44_z2_44_3ori_regression_model_lr0.001_epochs50_batch16_folds5.h5`
- **Training Data Collection**: 
  - Both magnets are placed at the same height (**44.27mm**).
  - **Three orientations** per magnet are used for training.
- **Scaler**: `scaler_regression_z1_44_z2_44_1ori.joblib`

### 3. `2mag_sameH_44546474_1ori_regression_model_lr0.001_epochs200_batch16_folds5.h5`
- **Training Data Collection**:
  - Both magnets are placed at same height, but at multiple heights (**44.27mm**, **54.27mm**, **64.27mm**, **74.27mm**).
  - Only **one orientation** per magnet is used for training.
- **Scaler**: `scaler_regression_2mag_sameH_44546474_1ori.joblib`

## Important Notes
- These models are **less accurate** compared to the single magnet prediction model.
- Before executing the codes, there should be no magnet on the plate. When the codes start running, the magnet can be placed. (reason: if there is magnet at the beginning, the background data is wrong)

## How to Execute the Code

1. Navigate to the project directory:
   $ cd OneDrive/desktop/Peaba workspace/Permatracks/sinple_NN_regression_multi_magnets/
2. uncomment the model and corresponding scaler you want to test
3. Run the localization script:
   $ python localization_NN_regression_multi_mag.py


