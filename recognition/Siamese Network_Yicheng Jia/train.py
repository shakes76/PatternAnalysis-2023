import os
import random
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import dataset
from modules import SiameseNetwork


def check_CUDA():
    # Check if CUDA is available and set the device accordingly
    current_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
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
    criterion_function = nn.BCELoss()
    return criterion_function


def get_optimizer(net):
    # Define the optimizer using Adam
    train_optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=0.01)
    return train_optimizer


def instantiate_datasets_and_dataloaders():
    # Instantiate the datasets and dataloaders
    new_train_dataset, new_test_dataset, new_val_dataset = dataset.get_datasets('AD_NC')

    new_train_dataloader = DataLoader(new_train_dataset, shuffle=True, num_workers=4, batch_size=256)
    new_test_dataloader = DataLoader(new_test_dataset, shuffle=True, num_workers=4, batch_size=256)
    new_validation_dataloader = DataLoader(new_val_dataset, shuffle=True, num_workers=4, batch_size=256)

    return new_train_dataset, new_train_dataloader, new_test_dataloader, new_val_dataset, new_validation_dataloader


def validate_process():
    print("Start to check or create validating set...")
    # Define source and validation directories
    train_path = "AD_NC/train"
    validate_path = "AD_NC/val"

    # Check if validation set already exists
    if not os.path.exists(validate_path) or len(os.listdir(validate_path)) == 0:
        # get all patient ids
        all_patient_ids = set()
        for current_category in ["AD", "NC"]:
            for file_name in os.listdir(os.path.join(train_path, current_category)):
                patient_id = file_name.split('_')[0]  # get the part before the underscore as the patient id
                all_patient_ids.add(patient_id)

        # randomly choose 20% of the patients for validation
        number_of_validated_patients = int(0.2 * len(all_patient_ids))  # 20% of the patients
        validation_patient_ids = random.sample(list(all_patient_ids), number_of_validated_patients)  # randomly choose 20% of the patients

        # move the images of the validation patients to the validation directory
        for current_category in ["AD", "NC"]:
            for file_name in os.listdir(os.path.join(train_path, current_category)):
                patient_id = file_name.split('_')[0]  # get the part before the underscore as the patient id
                source_path = os.path.join(train_path, current_category, file_name)
                if patient_id in validation_patient_ids:
                    destination_path = os.path.join(validate_path, current_category, file_name)
                    if not os.path.exists(os.path.dirname(destination_path)):
                        os.makedirs(os.path.dirname(destination_path))
                    shutil.move(source_path, destination_path)

        # Print a message
        print("Validating set created successfully!")
    else:
        print("Validating set already exists!")


# def read_from_checkpoint(current_module_pth: str):
#     # read the module
#     current_checkpoint = torch.load(current_module_pth)
#     net.load_state_dict(current_checkpoint['module_state_dict'])
#     current_training_epoch = current_checkpoint['epoch']
#
#     return current_checkpoint, current_training_epoch


def train_loop_with_progress_bar(train_module, train_epoch, train_loop_dataloader, train_optimizer):
    print("Main training loop is in progress...")

    train_module.train()
    train_epoch = train_epoch if train_epoch is not None else 0
    # Initialize the running loss to 0
    train_total_losses = 0.0
    train_correct_numbers = 0
    train_total_labels = 0
    # Progress bar
    for index, image_data in tqdm(enumerate(train_loop_dataloader), total=len(train_loop_dataloader)):
        # Get the images and labels from the data
        img_0, img_1, image_label = image_data
        # Move the images and labels to the appropriate device
        img_0, img_1, image_label = img_0.to(device), img_1.to(device), image_label.to(device)

        # Zero the gradients
        train_optimizer.zero_grad()
        total_train_outputs = train_module(img_0, img_1).squeeze()  # Adjust the size of the output to match the target
        train_loss = criterion(total_train_outputs, image_label)
        train_loss.backward()
        train_optimizer.step()

        # Calculate the accuracy
        predicted_label = torch.where(total_train_outputs > 0.5, torch.tensor(1, device=device), torch.tensor(0, device=device))
        train_total_losses += train_loss.item()
        train_total_labels += len(image_label)
        train_correct_numbers += predicted_label.eq(image_label.view_as(predicted_label)).sum().item()

    # Calculate the average loss over the entire training data
    average_train_loss = train_total_losses / len(train_loop_dataloader)
    average_train_accuracy = 100 * train_correct_numbers / train_total_labels
    print(f"Training - Epoch: {train_epoch}, Training Average Loss: {average_train_loss:.4f}, Training Accuracy: {average_train_accuracy:.4f}%")

    # Write the average loss value to tensorboard
    writer.add_scalar('Training Average Loss', average_train_loss, train_epoch)
    writer.add_scalar('Training Accuracy', average_train_accuracy, train_epoch)

    train_content = "Training_Epoch:" + str(train_epoch) + ",Training_Average_Loss:" + str(average_train_loss) + ",Training_Accuracy:" + str(average_train_accuracy)
    write_to_file(file_path, train_content)


def validation_loop_with_progress_bar(validate_module, epoch, validate_loop_dataloader):
    # Validation loop
    validate_module.eval()

    print("Validating is in progress...")
    validate_total_loss = 0.0
    validate_accuracy = 0.0
    validate_total_labels = 0.0
    # Progress bar
    for index, image_data in tqdm(enumerate(validate_loop_dataloader), total=len(validate_loop_dataloader)):
        img_0, img_1, image_label = image_data
        img_0, img_1, image_label = img_0.to(device), img_1.to(device), image_label.to(device)
        with torch.no_grad():
            validate_total_outputs = validate_module(img_0, img_1).squeeze()
            validate_loss = criterion(validate_total_outputs, image_label)

            predicted_label = torch.where(validate_total_outputs > 0.5, torch.tensor(1, device=device), torch.tensor(0, device=device))
            validate_total_labels += len(image_label)
            validate_accuracy += predicted_label.eq(image_label.view_as(predicted_label)).sum().item()
            validate_total_loss += validate_loss.sum().item()

    average_validation_loss = validate_total_loss / len(validate_loop_dataloader)
    validation_accuracy = validate_accuracy / validate_total_labels * 100
    print(f"Validating - Epoch: {epoch}, Validating Average Loss: {average_validation_loss:.4f},Validating Accuracy:{validation_accuracy:.4f}%")
    # Write the average validation loss value and accuracy to tensorboard
    writer.add_scalar('Validation Average Loss', average_validation_loss, epoch)
    writer.add_scalar('Validation Accuracy', validation_accuracy, epoch)
    current_content = "Validating_Average_Loss:" + str(average_validation_loss) + ",Validating_Accuracy:" + str(validation_accuracy)
    write_to_file(file_path, current_content)

    return validation_accuracy, average_validation_loss


def test_loop_with_progress_bar(test_module, epoch, test_loop_dataloader):
    # Test Loop
    test_module.eval()
    print("Testing is in progress...")
    test_total_loss = 0.0
    test_accuracy = 0.0
    test_total_labels = 0.0
    # Progress bar
    for index, image_data in tqdm(enumerate(test_loop_dataloader), total=len(test_loop_dataloader)):
        img_0, img_1, image_label = image_data
        img_0, img_1, image_label = img_0.to(device), img_1.to(device), image_label.to(device)
        with torch.no_grad():
            test_total_outputs = test_module(img_0, img_1).squeeze()
            test_loss = criterion(test_total_outputs, image_label)
            predicted_label = torch.where(test_total_outputs > 0.5, torch.tensor(1, device=device), torch.tensor(0, device=device))
            test_total_labels += len(image_label)
            test_accuracy += predicted_label.eq(image_label.view_as(predicted_label)).sum().item()
            test_total_loss += test_loss.sum().item()

    test_average_loss = test_total_loss / len(test_loop_dataloader)
    test_accuracy = test_accuracy / test_total_labels * 100
    print(f"Testing - Epoch:{epoch},Testing_Average_Loss:{test_average_loss:.4f},Testing_Accuracy:{test_accuracy:.4f}%")

    # Write the average test loss value and accuracy to tensorboard
    writer.add_scalar('Test Average Loss', test_average_loss, epoch)
    writer.add_scalar('Test Accuracy', test_accuracy, epoch)
    current_content = "Testing_Average_Loss:" + str(test_average_loss) + ",Testing_Accuracy:" + str(test_accuracy)
    write_to_file(file_path, current_content)

    return test_accuracy, test_average_loss


def main_training_loop(current_module, epochs, current_train_dataloader, current_test_dataloader, current_validation_dataloader, train_optimizer):
    # Main Loop
    current_module.train()
    best_validate_accuracy = 0.0
    best_validate_loss = float('inf')
    best_test_accuracy = 0.0
    best_test_loss = float('inf')

    for epoch in range(epochs + 1, 1001):
        train_loop_with_progress_bar(current_module, epoch, current_train_dataloader, train_optimizer)
        validate_accuracy, validate_loss = validation_loop_with_progress_bar(current_module, epoch, current_validation_dataloader)

        test_accuracy, test_loss = test_loop_with_progress_bar(current_module, epoch, current_test_dataloader)

        if test_accuracy > best_test_accuracy or test_loss < best_test_loss:
            print(f"Current test accuracy: {test_accuracy:.4f}%, Current test loss: {test_loss:.4f}")
            best_test_accuracy = test_accuracy
            best_test_loss = test_loss
            best_content = "Best_Testing_Average_Loss:" + str(best_test_loss) + ", Best_Testing_Accuracy:" + str(best_test_accuracy)
            write_to_file(file_path, best_content)
            torch.save({
                'module_state_dict': current_module.state_dict(),
                'epoch': epoch,
            }, 'module_best_test_accuracy' + '.pth')

        if validate_accuracy > best_validate_accuracy or validate_loss < best_validate_loss:
            print(f"Current validating accuracy: {validate_accuracy:.4f}%, Current validating loss: {validate_loss:.4f}")
            best_validate_accuracy = validate_accuracy
            best_validate_loss = validate_loss
            best_content = "Best_Validating_Average_Loss:" + str(best_validate_loss) + ", Best_Validating_Accuracy:" + str(best_validate_accuracy)
            write_to_file(file_path, best_content)
            torch.save({
                'module_state_dict': current_module.state_dict(),
                'epoch': epoch,
            }, 'module_best_validation' + '.pth')


def write_to_file(file_path_to_save, content):
    with open(file_path_to_save, 'a') as file:
        file.write(content + '\n')


if __name__ == '__main__':
    # check whether CUDA is available
    device = check_CUDA()

    # define training log path and initialize tensorboard writer
    file_path, writer = training_log_path_and_writer()

    # Initialize the Siamese network and move it to the appropriate device
    module = SiameseNetwork().to(device)

    # Define the loss function
    criterion = loss_function()

    # Define the optimizer
    optimizer = get_optimizer(module)

    # Validation set check, create it if it doesn't exist
    validate_process()

    # Instantiate datasets and dataloaders
    train_dataset, train_dataloader, test_dataloader, val_dataset, validation_dataloader = instantiate_datasets_and_dataloaders()

    # read the module from the checkpoint of the previous module
    # checkpoint, current_epoch = read_from_checkpoint('module_best_test_accuracy.pth')
    current_epoch = 0
    main_training_loop(module, current_epoch, train_dataloader, test_dataloader, validation_dataloader, optimizer)

    writer.close()
