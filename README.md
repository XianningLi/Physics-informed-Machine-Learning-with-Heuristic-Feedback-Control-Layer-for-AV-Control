# Physics-informed-Machine-Learning-with-Heuristic-Feedback-Control-Layer-for-AV-Control
Physics-informed Machine Learning with Heuristic Feedback Control Layer for Autonomous Vehicle Control.

## Table of Contents

- [Project Structure](#project-structure)
- [Controllers](#controllers)
- [Key Features](#key-features)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Results](#results)
- [Author](#author)

---

## Project Structure

```plaintext
.
├── DPCModel/                 # DPC controller files
├── HFAMPCModel/              # HFAMPC controller files
├── HFRPCModel/               # HFRPC controller files
├── RPCModel/                 # RPC controller files
├── simulation/               # Simulation scripts
├── DataSetGeneration/        # Scripts to generate training and testing data
├── results/                  # Output results and performance summaries
├── mainLaneChangingSixController.py  # Main script for testing all controllers
├── mainNumericalSimulationSingleExample.py  # Script for a single example simulation
└── PlotLoss/                 # Training loss visualization files
```

---

## Controllers

This project implements the following controllers:
1. **MPC (Model Predictive Control)**
2. **AMPC (Approximate MPC)**
3. **HFAMPC (Heuristic Feedback AMPC)**
4. **DPC (Differentiable Predictive Control)**
5. **RPC (Recurrent Predictive Control)**
7. **HFRPC (Heuristic Feedback RPC)**

---

## Key Features

- **Physics-Informed Learning**: Combines machine learning with model-based control for enhanced generalization.
- **Heuristic Feedback Layer**: Improves steady-state error and generalization.
- **Performance Comparison**: Numerical and graphical analysis of computational efficiency, trajectory accuracy, and generalization capabilities.

---

## Dependencies

To run the project, the following dependencies are required:
- Python 3
- numpy
- pandas
- matplotlib
- torch
- casadi
- CUDA

---

## Usage

1. **Generate Dataset**:
   Use `DataSetGeneration/` scripts to generate training and testing data.
   You can also download the dataset using the following link:  
[Google Drive Dataset](https://drive.google.com/drive/folders/1M-Q4PIhni7Qef-Y_Tv5yNguGOi1WrRkw?usp=drive_link)


3. **Train Controllers**:
   Run training scripts in the respective model directories (e.g., `RPC_train_GPU.py`).

4. **Run Simulations**:
   - Single example simulation: `mainNumericalSimulationSingleExample.py`
   - Multiple controller comparison: `mainLaneChangingSixController.py`

5. **Visualize Results**:
   - Training loss: `Training Loss over Epochs.png`.

---

## Results

- **Performance Analysis**:
  - Performance comparisons are shown in `Lane-Change Instance Within Training Set Boundaries.png` and `Closed-loop Numerical Simulation Results.png`.
- **Training Loss**:
  - Visualized in `Training Loss over Epochs.png`.

---

## Author
Xianning Li  
New York University  
Email: xl5305@nyu.edu
