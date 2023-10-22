"""
Task 8 OASIS Brain image generation constants
Ryan Ward
45813685
"""

"""Default Hyperparameters"""
BATCH_SIZE = 32

HIDDEN_LAYERS = 128
RESIDUAL_HIDDEN_LAYERS = 32
RESIDUAL_LAYERS = 2

EMBEDDING_DIMENSION = 128
EMBEDDINGS = 512

"""Taken from the original paper"""
BETA = 0.25

LEARNING_RATE = 1e-3
EPOCHS = 2

"""Training Data Paths"""
TRAIN_PATH = "./keras_png_slices_data/keras_png_slices_train"
TEST_PATH  = "./keras_png_slices_data/keras_png_slices_test"
VAL_PATH   = "./keras_png_slices_data/keras_png_slices_validate"


