"""
parameters.py

Parameters for the Perceiver transformer.

Author: Atharva Gupta
Date Created: 17-10-2023
"""

# Data path and training parameters
DATA_PATH = "C:/Users/hp/Desktop/comp3710/PatternAnalysis-2023/recognition/s48195609/AD_NC"
EPOCHS = 10  # Number of training epochs
BATCH_SIZE = 16  # Batch size for training
LEARNING_RATE = 0.001  # Learning rate for the optimizer

# Model parameters
NUM_LATENTS = 32  # Number of latent units
DIM_LATENTS = 128  # Dimension of the latent units
NUM_CROSS_ATTENDS = 1  # Number of cross-attention layers
DEPTH_LATENT_TRANSFORMER = 4  # Depth of the latent transformer

# Model save path
MODEL_PATH = 'PatternAnalysis-2023/perceiver_transformer'  # Path to save the trained model
