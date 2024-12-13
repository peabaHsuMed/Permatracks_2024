# Prediction the position of 2 magnets with ResNet20 Model

This folder contains the implementation of using a ResNet20 model for predicting the positions of **two magnets**. The models are trained using data collected on the old grid plate (spacing: 30mm).

## Models Available for Testing

### 1. `ResNet20_model_superposition_height_44_292118_samples_3ori_2mag.keras`
- **Training Data Collection**: 
  - Both magnets are placed at the same height (**44.27mm**).
  - Only **three orientations** per magnet is used for training.
- **Scaler**: `scaler_ResNet20_superposition_z44_3ori_292118samples_x.joblib`, `scaler_ResNet20_superposition_z44_3ori_292118samples_x.joblib`, `scaler_ResNet20_superposition_z44_3ori_292118samples_x.joblib`

## How to Execute the Code

1. Navigate to the project directory:
   $ cd OneDrive/desktop/Peaba workspace/Permatracks/ResNet20_multi_magnets/
3. Run the localization script:
   $ python localizatio_ResNet20_multi_mag.py


