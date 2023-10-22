import glob
from sklearn.utils import shuffle

# Define paths
DATA_PATH = 'C:/Users/keefe/Documents/COMP3710/Task1/DATA/'
TRAIN_PATH = '/ISIC2018_Task1-2_Training_Input_x2/'
MASK_PATH = '/ISIC2018_Task1_Training_GroundTruth_x2/'


def load_data():
    # Load data from Dataset
    img_paths = sorted(glob.glob(DATA_PATH + TRAIN_PATH + '/*.jpg'))
    mask_paths = sorted(glob.glob(DATA_PATH + MASK_PATH + '/*.png'))

    # Shuffle data to randomise / prevent overfitting
    img_paths, mask_paths = shuffle(img_paths, mask_paths)
