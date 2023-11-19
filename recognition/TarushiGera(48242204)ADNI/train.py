"""
Swin Transformer Model Architecture-Based Topic Recognition for Alzheimer's Disease Classification
Name: Tarushi Gera
Student ID: 48242204
This script thoroughs a channel to train and test a Swin Transformer Model on the Alzheimer's Dataset, with the advantage of cropping parameters through a YAML configuration file.
"""
import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset import AlzheimerDataset  # Import your dataset class from dataset.py
from modules import SwinTransformer  # Import your Swin Transformer model from modules.py
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

# Define Trainer class and initialize it 
class Trainer:
    def __init__(self, config):
        
        self.is_train = config['training']['is_train']
        self.is_test = config['testing']['is_test']
        
        self.train_data_dir = config['data']['train_data_dir']
        self.val_data_dir = config['data']['val_data_dir']
        self.test_data_dir = config['data']['test_data_dir']
        
        self.device = config['training']['device']
        self.num_classes = config['model']['num_classes']
        
        self.image_size = config['training']['image_size']
        self.batch_size = config['training']['batch_size']
        self.learning_rate = float(config['training']['learning_rate'])
        self.epochs = config['training']['epochs']
        self.save_dir = config['training']['save_dir']
        os.makedirs(self.save_dir, exist_ok=True)

        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),  # Resize images to a fixed size
            transforms.ToTensor(),           # Convert to tensor
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize pixel values
        ])  
        
        
        if self.is_train:
            self.train_dataset = AlzheimerDataset(self.train_data_dir, transform=self.transform)
            self.val_dataset = AlzheimerDataset(self.val_data_dir, transform=self.transform)
        
            self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
            self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size)
        
        if self.is_test:
            self.test_dataset = AlzheimerDataset(self.test_data_dir, transform=self.transform)
            self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size)

        self.model = SwinTransformer(
            image_size=self.image_size[0], 
            num_classes=self.num_classes)

        if config['testing']['model_path']:
            checkpoint = torch.load(config['testing']['model_path'])
            self.model.load_state_dict(checkpoint)
        
        self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=0.005)

        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        print(len(self.test_loader))

    def train(self):
        print("\n\t\t\tTraining Started\n")
        previous_best = 0
        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0
            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            # Calculate average training loss for this epoch
            avg_train_loss = running_loss / len(self.train_loader)
            self.train_losses.append(avg_train_loss)

            # Validation loop
            self.model.eval()
            correct = 0
            total = 0
            val_loss = 0.0
            with torch.no_grad():
                for inputs, labels in self.val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    val_loss += loss.item()

                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            # Calculate validation loss and accuracy for this epoch
            avg_val_loss = val_loss / len(self.val_loader)
            self.val_losses.append(avg_val_loss)
            val_accuracy = 100 * correct / total
            self.val_accuracies.append(val_accuracy)

            print(f"Epoch [{epoch + 1}/{self.epochs}] - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f} - Val Acc: {val_accuracy:.2f}%")
            
            if val_accuracy > previous_best:
                self.save_model(f'model_{epoch+1}.pth')
                previous_best = val_accuracy

    def plot_losses(self):
        # Plot training and validation losses
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label="Train Loss")
        plt.plot(self.val_losses, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Training and Validation Losses")
        plt.savefig(f"{self.save_dir}/loss_plot.png")

    def plot_val_accuracy(self):
        # Plot validation accuracy
        plt.figure(figsize=(10, 5))
        plt.plot(self.val_accuracies, label="Val Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.legend()
        plt.title("Validation Accuracy")
        plt.savefig(f"{self.save_dir}/accuracy_plot.png")

    def save_model(self, model_path="swin_transformer_model.pth"):
        print("Saving Model")
        torch.save(self.model.state_dict(), self.save_dir+'/'+model_path)

    def test(self):
        print("\n\t\t\tTesting Started\n")
        self.model.eval()
        correct = 0
        total = 0
        all_labels = []
        all_predictions = []
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

        test_accuracy = 100 * correct / total
        print(f"Test Accuracy: {test_accuracy:.2f}%")
        
        # Compute the confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)

        # Make it into a pretty plot
        fig, ax = plt.subplots(figsize=(9, 7))
        ax.set_title("Fashion Confusion Matrix")
        cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['0','1'])
        cm_display.plot(ax=ax, cmap="Blues", colorbar=False)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.savefig(f"{self.save_dir}/cm_plot.png")

        
    def run(self):
        if self.is_train:
            self.train()
            self.plot_losses()
            self.plot_val_accuracy()
        
        if self.is_test:
            self.test()

if __name__ == "__main__":
    
    # Load the configuration from config.yaml
    with open("config.yaml", "r") as config_file:
        config = yaml.safe_load(config_file)

    trainer = Trainer(config)
    trainer.run()
