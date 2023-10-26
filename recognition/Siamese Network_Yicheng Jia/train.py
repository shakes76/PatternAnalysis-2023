"""
    File name: train.py
    Author: Yicheng Jia
    Date created: 27/09/2023
    Date last modified: 26/10/2023
    Python Version: 3.10.12
"""


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
    criterion_function = nn.BCELoss()
    return criterion_function


def get_optimizer(net):
    # Define the optimizer using Adam
    train_optimizer = optim.Adam(net.parameters(), lr=0.001)
    # train_optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=0.01)
    return train_optimizer


def instantiate_datasets_and_dataloaders():
    dataloader_batch_size = 256
    number_of_workers = 4
    # Instantiate the datasets and dataloaders
    new_train_dataset, new_test_dataset, new_val_dataset = dataset.get_datasets('AD_NC')

    new_train_dataloader = DataLoader(new_train_dataset, num_workers=number_of_workers, batch_size=dataloader_batch_size)
    new_test_dataloader = DataLoader(new_test_dataset, num_workers=number_of_workers, batch_size=dataloader_batch_size)
    new_validation_dataloader = DataLoader(new_val_dataset, num_workers=number_of_workers, batch_size=dataloader_batch_size)

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


def train_loop_with_progress_bar(train_module, train_epoch, train_loop_dataloader, train_optimizer):

    # Train loop
    train_module.train()
    print("Main training loop is in progress...")

    # Initial values
    train_epoch = train_epoch if train_epoch is not None else 0
    train_total_losses = 0.0
    train_correct_numbers = 0
    train_total_labels = 0

    # Progress bar
    for index, image_data in tqdm(enumerate(train_loop_dataloader), total=len(train_loop_dataloader)):

        img_0, img_1, image_label = image_data  # Get the images and labels from the data
        img_0, img_1, image_label = img_0.to(device), img_1.to(device), image_label.to(device)  # Move the images and labels to the appropriate device
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
    train_average_loss = train_total_losses / len(train_loop_dataloader)
    train_average_accuracy = 100. * train_correct_numbers / train_total_labels
    print(f"Training - Epoch: {train_epoch},[{train_total_labels}/{len(train_loop_dataloader.dataset)}] "
          f"Training Average Loss: {train_average_loss:.4f}, Training Accuracy: {train_average_accuracy:.4f}%, ")

    # Write the average loss value to tensorboard
    writer.add_scalar('Training Average Loss', train_average_loss, train_epoch)
    writer.add_scalar('Training Accuracy', train_average_accuracy, train_epoch)
    train_content = "Training_Epoch:" + str(train_epoch) + ",Training_Average_Loss:" + str(train_average_loss) + ",Training_Accuracy:" + str(train_average_accuracy)
    write_to_file(file_path, train_content)


def validation_loop_with_progress_bar(validate_module, epoch, validate_loop_dataloader):
    # Validation loop
    validate_module.eval()
    print("Validating is in progress...")

    # Initial values
    validate_total_losses = 0.0
    validate_correct_numbers = 0.0
    validate_total_labels = 0.0

    # Progress bar
    with torch.no_grad():
        for index, image_data in tqdm(enumerate(validate_loop_dataloader), total=len(validate_loop_dataloader)):

            img_0, img_1, image_label = image_data  # Get the images and labels from the data
            img_0, img_1, image_label = img_0.to(device), img_1.to(device), image_label.to(device)  # Move the images and labels to the appropriate device
            validate_total_outputs = validate_module(img_0, img_1).squeeze()  # Adjust the size of the output to match the target
            validate_average_loss = criterion(validate_total_outputs, image_label)  # Get the loss

            # Calculate the accuracy
            predicted_label = torch.where(validate_total_outputs > 0.5, torch.tensor(1, device=device), torch.tensor(0, device=device))
            validate_total_labels += len(image_label)
            validate_correct_numbers += predicted_label.eq(image_label.view_as(predicted_label)).sum().item()
            validate_total_losses += validate_average_loss.item()

    # Calculate the average loss over the entire validating data
    validate_average_loss = validate_total_losses / len(validate_loop_dataloader)
    validate_average_accuracy = 100. * validate_correct_numbers / validate_total_labels
    print(f"Validating - Epoch: {epoch},[{validate_total_labels}/{len(validate_loop_dataloader.dataset)}] "
          f"Validating Average Loss: {validate_average_loss:.4f},Validating Accuracy:{validate_average_accuracy:.4f}%")

    # Write the average validation loss value and accuracy to tensorboard
    writer.add_scalar('Validation Average Loss', validate_average_loss, epoch)
    writer.add_scalar('Validation Accuracy', validate_average_accuracy, epoch)
    validate_content = "Validating_Average_Loss:" + str(validate_average_loss) + ",Validating_Accuracy:" + str(validate_average_accuracy)
    write_to_file(file_path, validate_content)

    # Update the current accuracy and loss in order to store the best module
    return validate_average_accuracy, validate_average_loss


def test_loop_with_progress_bar(test_module, test_epoch, test_loop_dataloader):
    # Test Loop
    test_module.eval()
    print("Testing is in progress...")

    # Initial values
    test_total_losses = 0.0
    test_correct_numbers = 0.0
    test_total_labels = 0.0

    # Progress bar
    with torch.no_grad():
        for index, image_data in tqdm(enumerate(test_loop_dataloader), total=len(test_loop_dataloader)):

            img_0, img_1, image_label = image_data  # Get the images and labels from the data
            img_0, img_1, image_label = img_0.to(device), img_1.to(device), image_label.to(device)  # Move the images and labels to the appropriate device
            test_total_outputs = test_module(img_0, img_1).squeeze()  # Adjust the size of the output to match the target
            test_loss = criterion(test_total_outputs, image_label)  # Get the loss

            # Calculate the accuracy
            predicted_label = torch.where(test_total_outputs > 0.5, torch.tensor(1, device=device), torch.tensor(0, device=device))
            test_total_labels += len(image_label)
            test_correct_numbers += predicted_label.eq(image_label.view_as(predicted_label)).sum().item()
            test_total_losses += test_loss.item()

    # Calculate the average loss over the entire testing data
    test_average_loss = test_total_losses / len(test_loop_dataloader)
    test_average_accuracy = 100. * test_correct_numbers / test_total_labels
    print(f"Testing - Epoch:{test_epoch},{test_epoch},[{test_total_labels}/{len(test_loop_dataloader.dataset)}] "
          f"Testing_Average_Loss:{test_average_loss:.4f},Testing_Accuracy:{test_average_accuracy:.4f}%")

    # Write the average test loss value and accuracy to tensorboard
    writer.add_scalar('Test Average Loss', test_average_loss, test_epoch)
    writer.add_scalar('Test Accuracy', test_average_accuracy, test_epoch)
    test_content = "Testing_Average_Loss:" + str(test_average_loss) + ",Testing_Accuracy:" + str(test_average_accuracy)
    write_to_file(file_path, test_content)

    # Update the current accuracy and loss in order to store the best module
    return test_average_accuracy, test_average_loss


def main_training_loop(current_module, start_epoch, end_epoch, current_train_dataloader, current_test_dataloader, current_validation_dataloader, train_optimizer):
    # Main Loop, starts from train loop so set the status as train(it's not necessary although)
    current_module.train()

    # Initial values in order to get the best module
    best_validate_accuracy = 0.0
    best_validate_loss = float('inf')
    best_test_accuracy = 0.0
    best_test_loss = float('inf')

    for epoch in range(start_epoch + 1, end_epoch):

        # Train loop
        train_loop_with_progress_bar(current_module, epoch, current_train_dataloader, train_optimizer)

        # Validate loop
        validate_accuracy, validate_loss = validation_loop_with_progress_bar(current_module, epoch, current_validation_dataloader)

        # Test loop
        test_accuracy, test_loss = test_loop_with_progress_bar(current_module, epoch, current_test_dataloader)

        # Compare the test accuracies and store the best one
        if test_accuracy > best_test_accuracy:
            print(f"Best testing accuracy: {test_accuracy:.4f}%, Best testing loss: {test_loss:.4f}")
            best_test_accuracy = test_accuracy

            # Compare the test losses and store the best one
            if test_loss < best_test_loss:
                # Loss is smaller and accuracy is higher, update the best module
                best_test_loss = test_loss
                both_best_content = "Best_Testing_Average_Loss:" + str(best_test_loss) + ", Best_Testing_Accuracy:" + str(best_test_accuracy)
                write_to_file(file_path, both_best_content)
                torch.save({
                    'module_state_dict': current_module.state_dict(),
                    'epoch': epoch,
                }, 'module_best_test_accuracy' + '.pth')
            else:
                # Loss is not smaller but accuracy is higher, still update the best module since we care about accuracy more
                best_content = " Best_Testing_Accuracy:" + str(best_test_accuracy)
                write_to_file(file_path, best_content)
                torch.save({
                    'module_state_dict': current_module.state_dict(),
                    'epoch': epoch,
                }, 'module_best_test_accuracy' + '.pth')

        # Compare the validating losses and accuracies, also store them and save as the best module
        if validate_accuracy > best_validate_accuracy and validate_loss < best_validate_loss:
            print(f"Best validating accuracy: {validate_accuracy:.4f}%, Best validating loss: {validate_loss:.4f}")
            best_validate_accuracy = validate_accuracy
            best_validate_loss = validate_loss
            best_content = "Best_Validating_Average_Loss:" + str(best_validate_loss) + ", Best_Validating_Accuracy:" + str(best_validate_accuracy)
            write_to_file(file_path, best_content)
            torch.save({
                'module_state_dict': current_module.state_dict(),
                'epoch': epoch,
            }, 'module_best_validation' + '.pth')


# Write the module to file
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
    train_dataset, train_dataloader, test_dataloader, validate_dataset, validate_dataloader = instantiate_datasets_and_dataloaders()

    # Initialize the start/end epoch, if we need to start from a checkpoint, we can modify it and use a helper function. 
    # I deleted them since there is no need to use.
    starting_epoch = 0
    ending_epoch = 201
    main_training_loop(module, starting_epoch, ending_epoch, train_dataloader, test_dataloader, validate_dataloader, optimizer)

    # Close the writer
    writer.close()
