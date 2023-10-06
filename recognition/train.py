from modules import get_maskrcnn_model
from dataset import SkinLesionDataset
import os

if __name__ == "__main__":
    train_set = SkinLesionDataset()
    train_set.load_dataset(dataset_dir="E:/comp3710/ISIC2018")
    train_set.prepare()

    model = get_maskrcnn_model()

    model.train(train_set, train_set, learning_rate=0.005, epochs=5, layers='heads')
