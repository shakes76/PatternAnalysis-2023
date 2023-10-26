import torch
import os
import argparse
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from dataset import load_data, create_data_loaders,load_test_data
from dataset import ADNC_Dataset, get_image_paths_from_directory, extract_patient_id
from torchvision import transforms
from torch.utils.data import DataLoader
from modules import ConvolutionalVisionTransformer, CViTConfig,configuration

def train_model(train_images_paths_AD, train_images_paths_NC, batch_size, model_save_path, num_epochs=10, learning_rate=0.001, num_classes=2, plot_path='plot'):
    """
    Trains CViT for image classification, designed for distinguishing between two classes, (AD) and (NC). The function encompasses the full pipeline from loading the data, 
    training the model through specified epochs, and validating the model's performance, to saving the trained model and visualizing the training process statistics.
    """
    # Load data
    train_images_AD, train_images_NC, val_images_AD, val_images_NC = load_data(train_images_paths_AD, train_images_paths_NC)

    # Create data loaders
    train_dataloader, val_dataloader = create_data_loaders(train_images_AD, train_images_NC, val_images_AD, val_images_NC, batch_size)

    config_params_dict=configuration()
    config = CViTConfig(config_params_dict)
    model = ConvolutionalVisionTransformer(config)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Lists to store training and validation statistics for each epoch
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_loss = 0.0
        correct_predictions = 0

        for images, labels in train_dataloader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            print(loss)
            print(outputs)
            print(labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            print(predicted)
            correct_predictions += (predicted == labels).sum().item()

        # Calculate average loss and accuracy for the epoch
        average_loss = total_loss / len(train_dataloader.dataset)
        accuracy = correct_predictions / len(train_dataloader.dataset)
        print(f'Epoch [{epoch + 1}/{num_epochs}] Train Loss: {average_loss:.4f} Train Accuracy: {accuracy:.4f}')

        # Validation phase
        model.eval()
        val_total_loss = 0.0
        val_correct_predictions = 0
        for val_images, val_labels in val_dataloader:
            with torch.no_grad():
                val_outputs = model(val_images)
                val_loss = criterion(val_outputs, val_labels)
                val_total_loss += val_loss.item()
                _, val_predicted = torch.max(val_outputs, 1)
                val_correct_predictions += (val_predicted == val_labels).sum().item()

        val_average_loss = val_total_loss / len(val_dataloader.dataset)
        val_accuracy = val_correct_predictions / len(val_dataloader.dataset)
        print(f'Epoch [{epoch + 1}/{num_epochs}] Validation Loss: {val_average_loss:.4f} Validation Accuracy: {val_accuracy:.4f}')
        # Save model and training stats in each epoch

        train_losses.append(average_loss)
        train_accuracies.append(accuracy)
        val_losses.append(val_average_loss)
        val_accuracies.append(val_accuracy)

    print('Training completed.')

    # You can also save the training statistics for future analysis
    training_stats = {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies
    }
    # Define the number of epochs
    num_epochs = len(training_stats['train_losses'])
    # Check if the folder exists before creating it
    if not os.path.exists(plot_path):
        try:
            os.makedirs(plot_path)
            print("Plot folder created successfully.")
        except OSError as e:
            print("Failed to create the path folder:", e)
    else:
        print("Plot folder already exists.")
    # Create a list of epoch numbers for x-axis
    epochs = list(range(1, num_epochs + 1))

    # Get training and validation losses and accuracies
    train_losses = training_stats['train_losses']
    val_losses = training_stats['val_losses']
    train_accuracies = training_stats['train_accuracies']
    val_accuracies = training_stats['val_accuracies']

    # Create subplots for loss and accuracy
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Plot loss
    ax1.plot(epochs, train_losses, label='Train Loss', marker='o')
    ax1.plot(epochs, val_losses, label='Validation Loss', marker='o')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Losses')
    ax1.legend()

    # Plot accuracy
    ax2.plot(epochs, train_accuracies, label='Train Accuracy', marker='o')
    ax2.plot(epochs, val_accuracies, label='Validation Accuracy', marker='o')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracies')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(f'{plot_path}/plot.png')

    torch.save(model.state_dict(), f'{model_save_path}')


def test_model(model_path, test_imagesAD_path, test_images_nc_path, batch_size):
    """
    Evaluate the performance of trained CViT model on a test dataset. The pretrained model and test dataset is loaded, and the model's
    classification accuracy on the test data is evaluated. 
    """
    # Load test data
    test_images_AD, test_images_NC = load_test_data(test_imagesAD_path, test_images_nc_path)

    # Create data loader for testing
    data_transforms = {
        'test': transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    test_dataset = ADNC_Dataset(test_images_AD, test_images_NC, transform=data_transforms['test'])
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    config_params_dict=configuration()
    config = CViTConfig(config_params_dict)
    model = ConvolutionalVisionTransformer(config)

    # Load the trained model weights
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Define a function to evaluate the model on the test set
    def evaluate_model(model, dataloader):
        model.eval()
        correct_predictions = 0
        total_samples = 0

        for images, labels in dataloader:
            with torch.no_grad():
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                correct_predictions += (predicted == labels).sum().item()
                total_samples += labels.size(0)

        accuracy = correct_predictions / len(dataloader.dataset)
        return accuracy

    # Evaluate the model on the test set
    test_accuracy = evaluate_model(model, test_dataloader)
    print(f'Test Accuracy: {test_accuracy:.4f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a model with specified parameters")

    parser.add_argument("--epoch", type=int, default=2, help="Number of epochs to train (default: 2)")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for training (default: 2)")

    args = parser.parse_args()

    train_images_paths_AD = "AD_NC/train/AD"
    train_images_paths_NC = "AD_NC/train/NC"
    test_images_paths_AD = "AD_NC/test/AD"
    test_images_paths_NC = "AD_NC/test/NC"
    model_save_path = "models.pth"
    save_plot = 'plot'
    learning_rate = 0.001
    num_classes = 2

    train_model(train_images_paths_AD, train_images_paths_NC, args.batch_size, model_save_path, args.epoch, learning_rate, num_classes, plot_path=save_plot)
    test_model(model_save_path, test_images_paths_AD, test_images_paths_NC, args.batch_size)
