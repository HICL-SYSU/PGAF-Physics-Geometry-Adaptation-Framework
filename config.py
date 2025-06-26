# config.py
import torch

# -----------------------------------------------------------------------------
#Basic Training Settings
# -----------------------------------------------------------------------------

# CUDA GPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# learning rate
LEARNING_RATE = 1e-4

# batch size
BATCH_SIZE = 1

# epochs
NUM_EPOCHS = 500


# -----------------------------------------------------------------------------
# Data & Path Settings
# -----------------------------------------------------------------------------

# data root: including train/test；  need to manual setting
DATA_DIR = "/data/Aorta-data"

# num works of data load
NUM_WORKERS = 4

# save path
MODEL_SAVE_PATH = "/results/saved_models/PGAF_model.pth"


# -----------------------------------------------------------------------------
# Model Architecture Parameters
# -----------------------------------------------------------------------------

# Channels of flow ( u, v, w，p -> 4)
FIELD_CHANNELS = 4

# Latent space dimension: the dimension of the latent variable z in the CVAE model
LATENT_DIM = 256

# Number of points to sample for boundary and query point clouds
# If set to None, will use all points (not recommended for varying point counts)
NUM_BOUNDARY_POINTS = 256
NUM_QUERY_POINTS = 32768      


# -----------------------------------------------------------------------------
# Loss Function Weights
# -----------------------------------------------------------------------------

# OT reconstruction loss weight: controls the degree to which the model fits the true data distribution
LAMBDA_OT = 1.0

# KLD loss weight (Beta for CVAE): controls the regularization strength of the latent space
# Smaller values (such as 0.1) make the model focus more on reconstruction, larger values (such as 1.0 or higher) force the latent space to be more regular
BETA_VAE = 1.0

# PDE physical loss weight: controls the strength of physical equation constraints
# The magnitude of PDE residuals is usually large, so this weight is usually set very small
LAMBDA_PDE = 1e-6


# -----------------------------------------------------------------------------
# Blood Properties
# -----------------------------------------------------------------------------
rho = 1060.0
mu = 0.004


# -----------------------------------------------------------------------------
# Early Stopping & Validation Settings
# -----------------------------------------------------------------------------

# Validation set ratio: How much data from the training set is used for validation
# 0.2 means 20% of the training data will be used as the validation set
VALIDATION_SPLIT_RATIO = 0.2

# Early stopping patience value: Stop training after how many epochs the validation loss has not improved
EARLY_STOPPING_PATIENCE = 20
