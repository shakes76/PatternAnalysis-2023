# Hyperparameters
IMAGE_SIZE = 128
PATCH_SIZE = 8
BATCH_SIZE = 16
PROJECTION_DIM = 16 # Depth of MLP blocks
LEARNING_RATE = 0.001
ATTENTION_HEADS = 5
DROPOUT_RATE = 0.2
TRANSFORMER_LAYERS = 5 # Number of transformer encoder blocks
WEIGHT_DECAY = 0.001
MLP_HEAD_UNITS = [256, 128]
##### AUTOMATICALLY CALCULATED #####
INPUT_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)
HIDDEN_UNITS = [PROJECTION_DIM * 2, PROJECTION_DIM]
NUM_PATCHES = int((IMAGE_SIZE/PATCH_SIZE) ** 2)
