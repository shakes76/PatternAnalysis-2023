"""
parameters.py

Hyperparameters for the visual transformer.

Author: Atharva Gupta
Date Created: 17-10-2023
"""

# Hyperparameters
IMAGE_SIZE = 56  # Reduce image size
PATCH_SIZE = 8
BATCH_SIZE = 16  # Reduce batch size
PROJECTION_DIM = 256  # Reduce projection dimension
LEARNING_RATE = 0.001  # Adjust learning rate accordingly
NUM_HEADS = 4  # Reduce the number of heads
DROPOUT_RATE = 0.1  # Adjust dropout rate
NUM_LAYERS = 4  # Reduce the number of layers
WEIGHT_DECAY = 0.0001
EPOCHS = 10
MLP_HEAD_UNITS = [128, 64]  # Reduce hidden units
LOCAL_WINDOW_SIZE = 8  # You can adjust this value as needed
DATA_LOAD_PATH = "C:/Users/hp/Desktop/comp3710/PatternAnalysis-2023/recognition/s48195609/AD_NC"
MODEL_SAVE_PATH = "C:/Users/hp/Desktop/comp3710/PatternAnalysis-2023/recognition/vision_transformer"

##### AUTOMATICALLY CALCULATED #####
INPUT_SHAPE = (3, IMAGE_SIZE, IMAGE_SIZE)  # Channels first for PyTorch
HIDDEN_UNITS = [PROJECTION_DIM * 2, PROJECTION_DIM]
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2
