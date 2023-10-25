import torch
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
import torchvision.transforms as tf
from torchvision.io import read_image, ImageReadMode
import matplotlib.pyplot as plt
import argparse
from utils import Config
from modules import Embedding_Baseline, ClassificationNet
from dataset import discover_directory, ClassificationDataset


# test acc
def test(model, val_loader):
    model_name = type(model).__name__
    model.to(Config.DEVICE)
    model.eval()
    total_acc = 0.0

    with torch.no_grad():
        for batch in tqdm(val_loader):
            imgs, labels = batch
            imgs, labels = imgs.to(Config.DEVICE), labels.to(Config.DEVICE)
            out = model(imgs)

            preds = (out > 0.5).float()
            corrects = (preds == labels).float().sum().cpu()
            batch_acc = corrects / len(labels)
            total_acc += batch_acc

    average_acc = total_acc / len(val_loader)

    # Print the information.
    print(f"[ test {model_name:}] acc = {average_acc:.5f}")

def predict_img(model, img_path, transform = None):
    model.eval()
    model.to(Config.DEVICE)
    label = img_path.split('/')[-2]
    img_name = img_path.split('/')[-1].split('.')[0]
    img = read_image(img_path, ImageReadMode.GRAY).float()/255.
    img.to(Config.DEVICE)
    if transform:
        img = transform(img)
    with torch.no_grad():
        out = model(img)
    pred = (out > 0.5).int().item()
    plt.imshow(img)
    plt.title(f"img: {img_name}, Truth: {label}, Prediction: {pred}")
    plt.axis('off')  # To not display axis values
    plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training script for Contrastive/Triplet network')

    # model
    parser.add_argument('-m', '--model', default='Contrastive', type=str,
                        help='model path to predict')
    parser.add_argument('-t', '--task', default='test', type=str,
                        help='task to do (test/patient_predict/image_predict')
    parser.add_argument('-d', '--data', default=None, type=str,
                        help='dataset to test or patient/image to predict')

    args = parser.parse_args()

    # model
    embedding_net = Embedding_Baseline()
    model = ClassificationNet(embedding_net)
    checkpoint = torch.load(args.model)
    model.load_state_dict(checkpoint['model_state_dict'])

    if args.task == 'test':

        # data
        val_transform = tf.Compose([
            tf.Normalize((0.1160,), (0.2261,))
        ])

        test_data = discover_directory(Config.TEST_DIR)
        test_dataset = ClassificationDataset(test_data, transform=val_transform)\

        test_loader = DataLoader(
            dataset=test_dataset,
            shuffle=True,
            batch_size=3,
            num_workers=1,
            drop_last=True
        )

        test(model, test_loader)
    elif args.task == 'image_predict':
        predict_img(model, args.data)
    #
    # elif args.task == 'patient_predict':







