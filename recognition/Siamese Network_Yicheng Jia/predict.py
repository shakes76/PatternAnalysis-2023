import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import dataset
from modules import SiameseNetwork


def write_to_file(file_path, content):
    with open(file_path, 'a') as file:
        file.write(content + '\n')


def check_CUDA():
    # Check if CUDA is available and set the device accordingly
    current_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return current_device


def read_from_checkpoint(current_model_pth: str):
    # read the model
    current_checkpoint = torch.load(current_model_pth)
    net.load_state_dict(current_checkpoint['module_state_dict'])
    current_training_epoch = current_checkpoint['epoch']

    return current_checkpoint, current_training_epoch


def predict_progress(network, current_device,  dataloader):
    network.eval()
    predicted_total_loss = 0.0
    predicted_accuracy = 0.0
    predicted_total_labels = 0.0
    print("Start testing!")
    # Progress bar
    for index, image_data in tqdm(enumerate(dataloader), total=len(dataloader)):
        img_0, img_1, image_label = image_data
        img_0, img_1, image_label = img_0.to(current_device), img_1.to(current_device), image_label.to(current_device)
        with torch.no_grad():
            predicted_total_outputs = net(img_0, img_1).squeeze()
            predicted_loss = criterion(predicted_total_outputs, image_label)

            predicted_label = torch.where(predicted_total_outputs > 0.5, torch.tensor(1, device=device), torch.tensor(0, device=device))
            predicted_total_labels += len(image_label)
            predicted_accuracy += predicted_label.eq(image_label.view_as(predicted_label)).sum().item()
            predicted_total_loss += predicted_loss.item()

    print("Testing finished, printing results...")

    predict_average_loss = predicted_total_loss / len(dataloader)
    accuracy = predicted_accuracy / predicted_total_labels * 100

    print(f"Test_Average_Loss:{predict_average_loss:.4f},Test_Accuracy:{accuracy:.4f}")


if __name__ == '__main__':

    device = check_CUDA()
    net = SiameseNetwork().to(device)

    # Define the loss function
    criterion = nn.BCELoss()
    ROOT_DIR = "AD_NC"
    # Create data loaders for training and testing datasets
    current_train_dataset, test_dataset = dataset.get_datasets(ROOT_DIR)
    test_dataloader = DataLoader(test_dataset, shuffle=True, num_workers=4, batch_size=256)

    # read the model from the checkpoint of the previous module
    checkpoint, current_epoch = read_from_checkpoint('module_best_test_accuracy.pth')

    predict_progress(net, device, test_dataloader)
