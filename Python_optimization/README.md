# Prediction of Single/Multiple Magnet Positions with Python Optimization Methods

This folder contains implementations of various optimization algorithms to retrieve the localization of magnets.

---

## Available Scripts

### 1. `position_retrieve_6DOF(matthew's ver).py`
- **Purpose**: Optimizes the position of a single magnet using 6 degrees of freedom.
- **Device**: Default for a 4x4 grid with 30mm spacing (can be adjusted for other devices by modifying the configurations).
- **Usage**: Single magnet optimization.

---

### 2. `opt_single_magnet_9DOF.py`
- **Purpose**: Optimizes the position of a single magnet using 9 degrees of freedom.
- **Device**: Default for a 4x4 grid with 30mm spacing (can be adjusted for other devices by modifying the configurations).
- **Experiments**: Uncomment the corresponding lines to try different configurations:
  - No calibration, no median filtering.
  - Median filtering only.
  - Calibration only.
  - Median filtering before calibration.
  - Calibration before median filtering.
- **Features**:
  - Plots the latency distribution of prediction time (uncomment the corresponding lines to enable this feature).

---

### 3. `opt_single_magnet_differential_evolution_and_dual_annealing.py`
- **Purpose**: Optimizes the position of a single magnet using **Differential Evolution** or **Dual Annealing** algorithms.
- **Device**: Default for a 4x4 grid with 30mm spacing (can be adjusted for other devices by modifying the configurations).
- **Options**:
  - Choose the optimization method by uncommenting the corresponding lines.
  - **Note**: Both methods are very slow and may seem inefficient for practical use.

---

### 4. `opt_loc_6M+3DOF_multi_magnet(cse).py`
- **Purpose**: Optimizes the position of multiple magnets using 6M+3 degrees of freedom.
- **Device**: Default for a 4x4 grid with 30mm spacing (can be adjusted for other devices by modifying the configurations).
- **Instructions**:
  - Adjust the `N_magnet` variable to specify the number of magnets.
  - Uncomment the corresponding lines for different numbers of magnets.
- **Features**:
  - Plots the latency distribution of prediction time (uncomment the corresponding lines to enable this feature).
- **Highlights**:
  - Implements a 6M+3 DOFs cost function and Jacobian, which is the **best-performing approach** for now.
  - Uses **Common Subexpression Elimination (CSE)** to slightly reduce computational time.
  - This approach has been ported to the C++ optimization version for improved performance.


## Notes
- All scripts are configurable for different devices by adjusting the respective configurations.
- Ensure the correct dependencies are installed for optimal performance.
- If tracking is lost, rotate the magnet(s) randomly to make algorithm re-track

## How to Execute the Code

1. Navigate to the project directory:
   $ cd OneDrive/desktop/Peaba workspace/Permatracks/Python_optimization
2. Run the localization script:
   $ python "desired_script"

