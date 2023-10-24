import torch
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
import torchvision.transforms as tf
import argparse
from utils import Config
from modules import Embedding_Baseline, ClassificationNet
from dataset import discover_directory, ClassificationDataset


# test acc
def test(model, val_loader):
    model_name = type(model).__name__
    model = model.to(Config.DEVICE)
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
    if args.task == 'test':
        # model
        embedding_net = Embedding_Baseline()
        embedding_net = embedding_net.to(Config.DEVICE)
        model = ClassificationNet(embedding_net)
        model = model.to(Config.DEVICE)
        checkpoint = torch.load(args.model, map_location=Config.DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])

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
    # elif args.task == 'image_predict':
    #
    # elif args.task == 'patient_predict':







