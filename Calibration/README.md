# Calibration of New Device (Sensor Board)

This folder contains the implementation for calibrating the sensor board.

## How to Execute the Calibration Code

1. Navigate to the project directory:
   $ cd OneDrive/desktop/Peaba workspace/Permatracks/Calibration
2. Update Configuration:
   Open calibration_ellip_retrieve.py
   Update the device's address and number of sensors in the script to match the new configuration.
3. Run the calibration script:
   $ python calibration_ellip_retrieve.py
4. Perform Calibration:
   Rotate the sensor board along its three axes (X, Y, Z).
   Perform two rotations per axis for sufficient data collection.
   Calibration will be completed automatically when no more data is printed in the terminal.
5. Save Calibration Coefficients
   The script will automatically save the calibration coefficients as: calibration_A_1.npy and calibration_b.npy
   Important: Rename the files after calibration to avoid confusion with future calibration runs.