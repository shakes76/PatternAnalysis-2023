from mrcnn.config import Config
from mrcnn import model as modellib

class MaskRCNNConfig(Config):
    NAME = "skin_lesion"
    IMAGES_PER_GPU = 16
    NUM_CLASSES = 1 + 1  # Background + skin lesion
    STEPS_PER_EPOCH = 100
    DETECTION_MIN_CONFIDENCE = 0.9


def get_maskrcnn_model(mode, config, model_dir, weights_path=None):
    """Returns a Mask R-CNN model.

    :param mode: Either 'training' or 'inference'
    :param config: A Subclass of mrcnn.config.Config
    :param model_dir: Directory to save logs and trained model
    :param weights_path: Path to pretrained weights
    :return: Mask R-CNN model
    """
    model = modellib.MaskRCNN(mode=mode, config=config, model_dir=model_dir)

    if weights_path:
        model.load_weights(weights_path, by_name=True)

    return model

if __name__ == '__main__':
    config = MaskRCNNConfig()
    config.display()
    model = get_maskrcnn_model("training", config, "E:/OneDrive/UQ/Year3_sem2/COMP3710/A3/logs")
    # ModuleNotFoundError: No module named 'keras.engine'
