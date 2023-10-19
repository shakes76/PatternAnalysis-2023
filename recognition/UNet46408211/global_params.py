# Global parameters for the UNet model defined here for easy access across files.

# Hyperparameters
LEARNING_RATE = 1e-4
BATCH_SIZE = 4
NUM_EPOCHS = 40
NUM_WORKERS = 1
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
PIN_MEMORY = True

# Downsampled data set paths
TRAIN_IMG_DIR = 'data/ISIC_2017_downsampled/train/images'
TRAIN_MASK_DIR = 'data/ISIC_2017_downsampled/train/masks'

TEST_IMG_DIR = 'data/ISIC_2017_downsampled/test/images'
TEST_MASK_DIR = 'data/ISIC_2017_downsampled/test/masks'

VAL_IMG_DIR = 'data/ISIC_2017_downsampled/val/images'
VAL_MASK_DIR = 'data/ISIC_2017_downsampled/val/masks'

# Original data set paths
# TRAIN_IMG_DIR = 'data/ISIC_2017/Training/ISIC-2017_Training_Data'
# TRAIN_MASK_DIR = 'data/ISIC_2017/Training/ISIC-2017_Training_Part1_GroundTruth'

# TEST_IMG_DIR = 'data/ISIC_2017/Testing/ISIC-2017_Test_v2_Data'
# TEST_MASK_DIR = 'data/ISIC_2017/Testing/ISIC-2017_Test_v2_Part1_GroundTruth'

# VAL_IMG_DIR = 'data/ISIC_2017/Validation/ISIC-2017_Validation_Data'
# VAL_MASK_DIR = 'data/ISIC_2017/Validation/ISIC-2017_Validation_Part1_GroundTruth'