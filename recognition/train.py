from modules import MaskRCNNConfig, get_maskrcnn_model
from dataset import load_training_set, load_val_set


def train_model(data_path, log_dir, pretrained_weights=None, epochs=10):
    """
    Train the Mask R-CNN model.

    :param data_path: Path to the dataset directory
    :param log_dir: Directory to save logs and trained model
    :param pretrained_weights: Path to the pretrained weights file
    :param epochs: Number of epochs for training
    """
    # Configuration
    config = MaskRCNNConfig()
    config.display()

    # Load datasets
    train_images, train_masks = load_training_set(data_path)
    val_images, val_masks = load_val_set(data_path)

    # Create model
    model = get_maskrcnn_model("training", config, log_dir)

    # Optionally, load pre-trained weights
    if pretrained_weights:
        model.load_weights(pretrained_weights, by_name=True,
                           exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])

    # Training
    model.train(train_images, train_masks,
                val_images, val_masks,
                learning_rate=config.LEARNING_RATE,
                epochs=epochs,
                layers='heads')  # 'heads' means that only the head layers of the network will be trained

    print("Training complete")


# Example usage
if __name__ == '__main__':
    data_path = "E:/comp3710/ISIC2018/"
    log_dir = "E:/OneDrive/UQ/Year3_sem2/COMP3710/A3/logs"
    train_model(data_path, log_dir, epochs=10)
