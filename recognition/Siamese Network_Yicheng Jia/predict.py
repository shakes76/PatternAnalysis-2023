import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import dataset
from modules import SiameseNetwork


def write_to_file(file_path, content):
    with open(file_path, 'a') as file:
        file.write(content + '\n')


def check_CUDA():
    # Check if CUDA is available and set the device accordingly
    if torch.cuda.is_available():
        current_device = torch.device("cuda")
        torch.cuda.empty_cache()
    else:
        current_device = torch.device("cpu")
    return current_device


def read_from_checkpoint(current_model_pth: str):
    # read the model
    current_checkpoint = torch.load(current_model_pth)
    net.load_state_dict(current_checkpoint['module_state_dict'])
    current_training_epoch = current_checkpoint['epoch']

    return current_checkpoint, current_training_epoch


def predict_progress(current_device,  dataloader):
    predicted_loss = 0.0
    predicted_accuracy = 0.0
    predicted_total_labels = 0.0
    print("Start testing!")
    for index, image_data in enumerate(dataloader):
        img_0, img_1, current_image_label = image_data
        img_0, img_1, current_image_label = img_0.to(current_device), img_1.to(current_device), current_image_label.to(current_device)
        with torch.no_grad():
            outputs = net(img_0, img_1)
            outputs = outputs.squeeze()
            loss_values = criterion(outputs, current_image_label)

        predicted_label = (outputs > 0.5).float()
        predicted_total_labels += current_image_label.size(0)
        predicted_accuracy += (predicted_label == current_image_label).sum().item()
        predicted_loss += loss_values.item()

    print("Testing finished, printing results...")

    test_average_loss = predicted_loss / len(dataloader)
    accuracy = predicted_accuracy / predicted_total_labels * 100

    print(f"Test_Average_Loss:{test_average_loss:.4f},Test_Accuracy:{accuracy:.4f}")


if __name__ == '__main__':

    device = check_CUDA()
    net = SiameseNetwork().to(device)

    # Define the loss function
    criterion = nn.BCELoss()
    net.eval()

    # Create data loaders for training and testing datasets AFTER re-instantiating train_dataset
    current_train_dir, test_dir, current_train_dataset, test_dataset = dataset.transform_directory()
    test_dataloader = DataLoader(test_dataset, shuffle=True, num_workers=4, batch_size=32)

    # read the model from the checkpoint of the previous module
    checkpoint, current_epoch = read_from_checkpoint('module_8.pth') # Please usethe pth file you prefre after running train.py.

    predict_progress(device, test_dataloader)
