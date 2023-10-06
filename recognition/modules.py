from tensorflow import keras
from mrcnn.config import Config
from mrcnn import model as modellib

class MaskRCNNConfig(Config):
    NAME = "skin_lesion"
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1  # Background + skin lesion
    STEPS_PER_EPOCH = 100
    DETECTION_MIN_CONFIDENCE = 0.9

def get_maskrcnn_model():
    config = MaskRCNNConfig()
    model = modellib.MaskRCNN(mode="training", config=config, model_dir="./")
    return model
