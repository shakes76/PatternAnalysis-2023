# Hyperparameters
IMAGE_SIZE = 128  # The size of the input image (in pixels)
PATCH_SIZE = 8  # The size of each image patch (in pixels)
BATCH_SIZE = 16  # The batch size used for training
PROJECTION_DIM = 128  # The depth of the MLP (Multi-Layer Perceptron) blocks
LEARNING_RATE = 0.001  # The initial learning rate for optimization
ATTENTION_HEADS = 5  # The number of attention heads in the multi-head self-attention mechanism
DROPOUT_RATE = 0.2  # The dropout rate used within the model
TRANSFORMER_LAYERS = 5  # The number of transformer encoder blocks in the architecture
WEIGHT_DECAY = 0.001  # The weight decay applied to model parameters during optimization
MLP_HEAD_UNITS = [256, 128]  # The sizes of the Multi-Layer Perceptron head layers

# Automatically Calculated Variables
INPUT_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)
# INPUT_SHAPE represents the shape of the input images, with dimensions (width, height, channels).

HIDDEN_UNITS = [PROJECTION_DIM * 2, PROJECTION_DIM]
# HIDDEN_UNITS represents the sizes of hidden units in the model, specifically in the MLP blocks.

NUM_PATCHES = int((IMAGE_SIZE / PATCH_SIZE) ** 2)
# NUM_PATCHES represents the total number of patches in the input image. It is computed by dividing the image width (IMAGE_SIZE) by the patch width (PATCH_SIZE) and squaring the result.
