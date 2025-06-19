# PGAF-Physics-Geometry-Adaptation-Framework

This project implements a deep learning model to predict blood flow dynamics (pressure and velocity fields) in complex geometries like the aorta and coronary arteries.


## Project Structure

```
.
├── data/                     # Main data directory
│   ├── train/                # Training data, split into 6 sub-folders
          ├── .../
          ...               
          ├── .../                
│   └── test/                
├── config.py                 # Central configuration file for all parameters
├── requirements.txt          # Python dependencies
├── mydata.py                 # Defines the custom PyTorch Geometric dataset
├── model.py                  # Contains the CVAE and Point-UNet architecture
├── loss_function.py          # Defines the hybrid loss function (OT + KLD + PDE)
├── train.py                  # Main script to start model training
├── predict.py                # Script to load a trained model and make predictions
└── README.md                 # This file
```

## Setup and Installation

### 1. Prerequisites
- Python 3.8+
- NVIDIA GPU with CUDA

### 2. Clone the Repository
```bash
git clone <your-repository-url>
cd <repository-name>
```

### 3. Install Dependencies

**Crucial Step**: PyTorch and PyTorch Geometric must be installed carefully to match your system's CUDA version.

**a. Install PyTorch:**

**b. Install PyTorch Geometric and its dependencies:**
Visit the [PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) to find the command that matches your PyTorch and CUDA version.
*Example for PyTorch 2.1.0 and CUDA 11.8:*
```bash
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
```

**c. Install remaining packages:**
```bash
pip install -r requirements.txt
```

Installation should take approximately 15-20 minutes on a standard machine with a good internet connection


## Data Preparation

The model expects data to be organized in a specific folder structure. Each sample (e.g., a patient case) should have its data split into 6 `.npy` files.

1.  Place your training data under `data/train/` and your test data under `data/test/`.
2.  The required sub-folder structure inside both `train/` and `test/` is:
    - `Inlet&Outlet_coords`: Coordinates of boundary points (shape: `(N_boundary, 3)`).
    - `Inlet&Outlet_flow`: Flow values at boundary points (shape: `(N_boundary, 4)` for p, u, v, w).
    - `Inter_coords`: Coordinates of interior points (shape: `(N_interior, 3)`).
    - `Inter_flow`: Flow values at interior points (ground truth) (shape: `(N_interior, 4)`).
    - `Wall_coords`: Coordinates of wall points (shape: `(N_wall, 3)`).
    - `Wall_flow`: Flow values at wall points (ground truth) (shape: `(N_wall, 4)`).

**Note:** For each sample, the corresponding files across the 6 folders must have the same name (e.g., `patient_001.npy`). 

## How to Run

### 1. Configure Your Training
Open `config.py` to adjust hyperparameters:
- `BATCH_SIZE`: Lower this if you run out of GPU memory.
- `LEARNING_RATE`: The optimizer's learning rate.
- `LAMBDA_OT`, `BETA_VAE`, `LAMBDA_PDE`: Weights for the different loss components.
- `VALIDATION_SPLIT_RATIO`: The fraction of training data to use for validation (e.g., `0.2` for 20%).

### 2. Train the Model
Simply run the training script from your terminal. The script will automatically detect the number of available GPUs.
```bash
python train.py
```
The training process will:
- Automatically split the `train` data into a training and validation set.
- Monitor validation loss and use Early Stopping to prevent overfitting.
- Save the model with the best validation performance to the path specified in `config.py` (default: `saved_models/`).

### 3. Make Predictions
After training is complete, run the prediction script to load the best model and visualize its output on a sample from the `test` set.
```bash
python predict.py
```
This will generate 3D plots comparing the ground truth flow field with the model's prediction.
