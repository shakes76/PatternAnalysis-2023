import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from train import Net  # Import the Net class from train.py
import matplotlib.pyplot as plt

model = Net()
model.load_state_dict(torch.load('model.pth'))
model.eval()

def load_and_preprocess_data(data_path):
    data = []
    filenames = []

    for filename in os.listdir(data_path):
        file_path = os.path.join(data_path, filename)
        if os.path.isfile(file_path) and file_path.endswith(('.jpg', '.png', '.jpeg')):
            image = Image.open(file_path).convert('RGB')
            transform = transforms.Compose([
                transforms.Resize((150, 150)),
                transforms.ToTensor(),
            ])
            image = transform(image)
            data.append(image)
            filenames.append(filename)

    data = torch.stack(data)

    return data, filenames

def plot_confidence_and_loss(train_data_path, test_data_path):
    train_data, train_filenames = load_and_preprocess_data(train_data_path)
    test_data, test_filenames = load_and_preprocess_data(test_data_path)

    if len(train_data) == 0 or len(test_data) == 0:
        print("No valid images found in the train or test directory.")
        return

    # Make predictions using the PyTorch model for training and testing data
    with torch.no_grad():
        train_outputs = model(train_data)
        test_outputs = model(test_data)

    # Ensure you have labels for calculating the loss (define labels accordingly)
    train_labels = torch.tensor([0, 1] * (len(train_filenames) // 2))  # Example labels, adjust as needed
    test_labels = torch.tensor([0, 1] * (len(test_filenames) // 2))  # Example labels, adjust as needed

    # Apply softmax to the model outputs for training and testing data
    train_softmax = nn.Softmax(dim=1)
    test_softmax = nn.Softmax(dim=1)
    train_predictions = train_softmax(train_outputs)
    test_predictions = test_softmax(test_outputs)

    # Calculate and display the final accuracy for training and testing data
    train_correct_predictions = 0
    test_correct_predictions = 0
    train_total_predictions = len(train_filenames)
    test_total_predictions = len(test_filenames)
    train_confidence_scores = []
    test_confidence_scores = []

    for i, train_prediction in enumerate(train_predictions):
        filename = train_filenames[i]
        true_label = filename.split("_")[0]
        predicted_label = 'AD' if train_prediction[0] > train_prediction[1] else 'NC'

        if true_label == predicted_label:
            train_correct_predictions += 1

        # Calculate confidence score for training data
        confidence = torch.max(train_prediction).item()
        train_confidence_scores.append(confidence)

    for i, test_prediction in enumerate(test_predictions):
        filename = test_filenames[i]
        true_label = filename.split("_")[0]
        predicted_label = 'AD' if test_prediction[0] > test_prediction[1] else 'NC'

        if true_label == predicted_label:
            test_correct_predictions += 1

        # Calculate confidence score for testing data
        confidence = torch.max(test_prediction).item()
        test_confidence_scores.append(confidence)

    train_accuracy = train_correct_predictions / train_total_predictions
    test_accuracy = test_correct_predictions / test_total_predictions

    print(f'Training Accuracy: {train_accuracy:.2f}')
    print(f'Testing Accuracy: {test_accuracy:.2f}')

    # Create and display bar charts for confidence scores for training and testing data
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.bar(train_filenames, train_confidence_scores, color='blue')
    plt.xlabel('Images')
    plt.ylabel('Confidence Score')
    plt.title('Training Confidence Scores')
    plt.grid(True)
    plt.xticks(rotation=45)

    plt.subplot(1, 2, 2)
    plt.bar(test_filenames, test_confidence_scores, color='blue')
    plt.xlabel('Images')
    plt.ylabel('Confidence Score')
    plt.title('Testing Confidence Scores')
    plt.grid(True)
    plt.xticks(rotation=45)

    # Save the confidence scores charts as images
    plt.savefig('training_confidence_scores.png', bbox_inches='tight')
    plt.savefig('testing_confidence_scores.png', bbox_inches='tight')

if __name__ == "__main":
    train_data_path = './recognition/48240983_ADNI/AD_NC/train'
    test_data_path = './recognition/48240983_ADNI/AD_NC/test'
    plot_confidence_and_loss(train_data_path, test_data_path)
