"""
parameter.py

Hyperparameters for the visual transformer.

Author: Atharva Gupta
Date Created: 17-10-2023
"""

# Hyperparameters
IMAGE_SIZE = 128
PATCH_SIZE = 8
BATCH_SIZE = 32
PROJECTION_DIM = 512  # Depth of MLP blocks
LEARNING_RATE = 0.0005
NUM_HEADS = 5  # Number of attention heads
DROPOUT_RATE = 0.2
NUM_LAYERS = 5  # Number of transformer encoder layers
WEIGHT_DECAY = 0.0001
EPOCHS = 10
MLP_HEAD_UNITS = [256, 128]
DATA_LOAD_PATH = "AD_NC"
MODEL_SAVE_PATH = "PATTERNANALYSIS/vision_transformer"

##### AUTOMATICALLY CALCULATED #####
INPUT_SHAPE = (3, IMAGE_SIZE, IMAGE_SIZE)  # Channels first for PyTorch
HIDDEN_UNITS = [PROJECTION_DIM * 2, PROJECTION_DIM]
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2