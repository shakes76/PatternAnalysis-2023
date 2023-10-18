import os
import random
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import ImageFolder
from tqdm import tqdm

import dataset
from dataset import SiameseNetworkDataset
from modules import SiameseNetwork


def check_CUDA():
    # Check if CUDA is available and set the device accordingly
    if torch.cuda.is_available():
        current_device = torch.device("cuda")
        torch.cuda.empty_cache()
    else:
        current_device = torch.device("cpu")
    return current_device


def training_log_path_and_writer():
    # write to file
    training_log_path = 'training_log.txt'
    # Initialize tensorboard writer
    summary_writer = SummaryWriter()
    return training_log_path, summary_writer


def loss_function():
    # Define the loss function
    # criterion = ContrastiveLoss()
    current_criterion = nn.BCELoss()
    return current_criterion


def current_optimizer():
    # Define the optimizer using Adam
    current_train_optimizer = optim.Adam(net.parameters(), lr=0.000002, weight_decay=0.01)
    return current_train_optimizer


def current_scheduler():
    # Define the learning rate scheduler
    current_train_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)
    return current_train_scheduler


def src_and_val_directories():
    # Define source and validation directories
    source_dir = "AD_NC/train"
    validation_dir = "AD_NC/val"
    return source_dir, validation_dir


def validate_process():
    print("Start to check or create validating set...")
    # Define source and validation directories
    current_train_path = "AD_NC/train"
    current_validate_path = "AD_NC/val"

    # Check if validation set already exists
    if not os.path.exists(current_validate_path) or len(os.listdir(current_validate_path)) == 0:
        # get all patient ids
        all_patient_ids = set()
        for current_category in ["AD", "NC"]:
            for file_name in os.listdir(os.path.join(current_train_path, current_category)):
                current_patient_id = file_name.split('_')[0]  # get the part before the underscore as the patient id
                all_patient_ids.add(current_patient_id)

        # randomly choose 20% of the patients for validation
        number_of_validated_patients = int(0.2 * len(all_patient_ids))  # 20% of the patients
        validation_patient_ids = random.sample(list(all_patient_ids), number_of_validated_patients)  # randomly choose 20% of the patients

        # move the images of the validation patients to the validation directory
        for current_category in ["AD", "NC"]:
            for file_name in os.listdir(os.path.join(current_train_path, current_category)):
                current_patient_id = file_name.split('_')[0]  # get the part before the underscore as the patient id
                source_path = os.path.join(current_train_path, current_category, file_name)
                if current_patient_id in validation_patient_ids:
                    destination_path = os.path.join(current_validate_path, current_category, file_name)
                    if not os.path.exists(os.path.dirname(destination_path)):
                        os.makedirs(os.path.dirname(destination_path))
                    shutil.move(source_path, destination_path)

        # Print a message
        print("Validating set created successfully!")
    else:
        print("Validating set already exists!")


def re_instantiate_and_create():
    transform = dataset.transform_process()
    current_train_dir, test_dir, current_train_dataset, test_dataset = dataset.transform_directory()
    _, validation_dir = src_and_val_directories()
    # Re-instantiate the train_dir and train_dataset after moving the images
    updated_train_dir = ImageFolder(root=os.path.join(os.getcwd(), "AD_NC/train"))
    updated_train_dataset = SiameseNetworkDataset(root_dir=os.path.join(os.getcwd(), "AD_NC/train"))

    # Create data loaders for training and testing datasets AFTER re-instantiating train_dataset
    new_train_dataloader = DataLoader(current_train_dataset, shuffle=True, num_workers=4, batch_size=32)
    new_test_dataloader = DataLoader(test_dataset, shuffle=True, num_workers=4, batch_size=32)

    # Create the validation dataset
    new_val_dataset = SiameseNetworkDataset(root_dir=validation_dir)

    # Create the validation data loader
    new_val_dataloader = DataLoader(new_val_dataset, shuffle=True, num_workers=4, batch_size=32)

    return updated_train_dir, updated_train_dataset, new_train_dataloader, new_test_dataloader, new_val_dataset, new_val_dataloader


def read_from_checkpoint(current_module_pth: str):
    # read the module
    current_checkpoint = torch.load(current_module_pth)
    net.load_state_dict(current_checkpoint['module_state_dict'])
    current_training_epoch = current_checkpoint['epoch']

    return current_checkpoint, current_training_epoch


def train_loop_with_progress_bar(current_module, train_epoch, current_train_dataloader, progress_optimizer):
    print("Main training loop is in progress...")
    current_net = current_module
    current_net.train()
    train_epoch = train_epoch if train_epoch is not None else 0
    # Initialize the running loss to 0
    current_running_loss = 0.0
    current_correct = 0
    current_total = 0
    # Progress bar
    for index, image_data in tqdm(enumerate(current_train_dataloader), total=len(current_train_dataloader)):
        # Get the images and labels from the data
        img_0, img_1, current_image_label = image_data
        # Move the images and labels to the appropriate device
        img_0, img_1, current_image_label = img_0.to(device), img_1.to(device), current_image_label.to(device)

        # Zero the gradients
        progress_optimizer.zero_grad()
        total_output = current_net(img_0, img_1)
        total_output = total_output.squeeze()  # Adjust the size of the output to match the target
        current_loss = criterion(total_output, current_image_label)
        current_loss.backward()
        progress_optimizer.step()

        # Calculate the accuracy
        predicted_label = (total_output > 0.5).float()
        current_running_loss += current_loss.item()
        current_total += current_image_label.size(0)
        current_correct += (predicted_label == current_image_label).sum().item()

    # Calculate the average loss over the entire training data
    average_loss = current_running_loss / len(current_train_dataloader)
    current_accuracy = 100 * current_correct / current_total
    print(f"Train - Epoch: {train_epoch}, Training Average Loss: {average_loss:.4f}, Training Accuracy: {current_accuracy:.2f}%")

    # Write the average loss value to tensorboard
    writer.add_scalar('Training Average Loss', average_loss, train_epoch)
    writer.add_scalar('Training Accuracy', current_accuracy, train_epoch)

    current_content = "Training_Epoch:" + str(train_epoch) + ",Training_Average_Loss:" + str(average_loss) + ",Training_Accuracy:" + str(current_accuracy)
    write_to_file(file_path, current_content)


def validation_loop(current_module, train_epoch, current_validation_dataloader):
    # Validation loop
    current_net = current_module
    current_net.eval()
    if train_epoch % 2 == 0:
        print("Validating is in progress...")
        torch.save({
            'epoch': train_epoch,
            'module_state_dict': net.state_dict(),
        }, 'module_' + str(train_epoch) + '.pth')

        validation_loss = 0.0
        current_accuracy = 0.0
        total_labels = 0.0
        for index, image_data in enumerate(current_validation_dataloader):
            img_0, img_1, current_image_label = image_data
            img_0, img_1, current_image_label = img_0.to(device), img_1.to(device), current_image_label.to(device)
            with torch.no_grad():
                total_output = net(img_0, img_1)
                total_output = total_output.squeeze()
                current_loss = criterion(total_output, current_image_label)

            predicted_label = (total_output > 0.5).float()
            total_labels += current_image_label.size(0)
            current_accuracy += (predicted_label == current_image_label).sum().item()
            validation_loss += current_loss.item()

        average_validation_loss = validation_loss / len(current_validation_dataloader)
        validation_accuracy = current_accuracy / total_labels * 100
        print(f"Validating - Epoch: {train_epoch}, Validating Average Loss: {average_validation_loss:.4f},Validating Accuracy:{validation_accuracy:4f}%")
        # Write the average validation loss value and accuracy to tensorboard
        writer.add_scalar('Validation Average Loss', average_validation_loss, train_epoch)
        writer.add_scalar('Validation Accuracy', validation_accuracy, train_epoch)
        current_content = "Validating_Average_Loss:" + str(average_validation_loss) + ",Validating_Accuracy:" + str(validation_accuracy)
        write_to_file(file_path, current_content)


def test_loop(current_module, train_epoch, current_test_dataloader):
    # Test Loop
    current_net = current_module
    current_net.eval()
    if train_epoch % 2 == 0:
        print("Testing is in progress...")
        current_test_loss = 0.0
        current_test_accuracy = 0.0
        current_test_label = 0.0
        for index, image_data in enumerate(current_test_dataloader):
            img_0, img_1, current_image_label = image_data
            img_0, img_1, current_image_label = img_0.to(device), img_1.to(device), current_image_label.to(device)
            with torch.no_grad():
                total_output = net(img_0, img_1)
                total_output = total_output.squeeze()
                current_loss = criterion(total_output, current_image_label)
            predicted_label = (total_output > 0.5).float()
            current_test_label += current_image_label.size(0)
            current_test_accuracy += (predicted_label == current_image_label).sum().item()
            current_test_loss += current_loss.item()

        test_average_loss = current_test_loss / len(current_test_dataloader)
        current_accuracy = current_test_accuracy / current_test_label * 100
        print(f"Testing - Epoch:{train_epoch},Testing_Average_Loss:{test_average_loss:.4f},Testing_Accuracy:{current_accuracy:.4f}%")
        # Write the average test loss value and accuracy to tensorboard
        writer.add_scalar('Test Average Loss', test_average_loss, train_epoch)
        writer.add_scalar('Test Accuracy', current_accuracy, train_epoch)
        current_content = "Testing_Average_Loss:" + str(test_average_loss) + ",Testing_Accuracy:" + str(current_accuracy)
        write_to_file(file_path, current_content)


def main_training_loop(current_module, epochs, current_train_dataloader, current_test_dataloader, current_validation_dataloader):
    # Main Loop
    current_net = current_module
    current_net.train()
    for train_epoch in range(epochs + 1, 21):
        train_loop_with_progress_bar(current_module, train_epoch, current_train_dataloader, optimizer)

        validation_loop(current_module, train_epoch, current_validation_dataloader)

        test_loop(current_module, train_epoch, current_test_dataloader)


def write_to_file(file_path_to_save, content):
    with open(file_path_to_save, 'a') as file:
        file.write(content + '\n')


if __name__ == '__main__':
    # check whether CUDA is available
    device = check_CUDA()

    # define training log path and initialize tensorboard writer
    file_path, writer = training_log_path_and_writer()

    # Initialize the Siamese network and move it to the appropriate device
    net = SiameseNetwork().to(device)

    # Define the loss function
    criterion = loss_function()

    # Define the optimizer using SGD
    optimizer = current_optimizer()

    # Define the learning rate scheduler, using StepLR
    scheduler = current_scheduler()

    # Define source and validation directories
    src_dir, val_dir = src_and_val_directories()

    # Validation set check, create it if it doesn't exist
    validate_process()

    # Re-instantiate train_dir and train_dataset, then create new datasets and new data loaders. 
    train_dir, train_dataset, train_dataloader, test_dataloader, val_dataset, validation_dataloader = re_instantiate_and_create()

    # read the module from the checkpoint of the previous module
    # checkpoint, current_epoch = read_from_checkpoint('model3_450.pth')
    current_epoch = 0
    main_training_loop(net, current_epoch, train_dataloader, test_dataloader, validation_dataloader)

    torch.save(net.state_dict(), "module_ful.pth")
    writer.close()

