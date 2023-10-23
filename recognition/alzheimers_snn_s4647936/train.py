import os
import torch
import matplotlib.pyplot as plt
from dataset import TripletDataset 
from modules import SiameseNetwork, TripletLoss, SimpleClassifier
from torchvision import transforms
import torch.optim as optim
from sklearn.manifold import TSNE
import numpy as np
import torch.nn as nn
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true, y_pred, classes, output_filename):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', cbar=False, ax=ax)
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(classes)
    ax.yaxis.set_ticklabels(classes)
    plt.savefig(output_filename)

# Transformations for images
transform = transforms.Compose([
    transforms.Resize((256, 240)),
    transforms.ToTensor(),
])

# Dataset instances
train_dataset = TripletDataset(root_dir="/home/Student/s4647936/PatternAnalysis-2023/recognition/alzheimers_snn_s4647936/AD_NC", mode='train', transform=transform)
test_dataset = TripletDataset(root_dir="/home/Student/s4647936/PatternAnalysis-2023/recognition/alzheimers_snn_s4647936/AD_NC", mode='test', transform=transform)

# Parameters
learning_rate = 0.001
num_epochs = 10
batch_size = 32

# GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if not torch.cuda.is_available():
    print("No CUDA Found. Using CPU")

# Initialise the Siamese Network and Triplet Loss
model = SiameseNetwork().to(device)
criterion = TripletLoss(margin=1.0).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# DataLoader setup
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# Lists to store training and validation losses
training_losses = []
validation_losses = []

# Training loop for Siamese Network
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch_idx, (anchor, positive, negative) in enumerate(train_loader):
        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        anchor_out, positive_out = model(anchor, positive)
        _, negative_out = model(anchor, negative)

        loss = criterion(anchor_out, positive_out, negative_out)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Calculate average training loss for the epoch
    epoch_loss = running_loss / len(train_loader)
    training_losses.append(epoch_loss)

    # Validation step
    model.eval() # set the model to evaluation mode
    val_running_loss = 0.0
    with torch.no_grad(): # deactivate autograd engine to reduce memory usage and speed up computations
        for val_anchor, val_positive, val_negative in test_loader:
            val_anchor, val_positive, val_negative = val_anchor.to(device), val_positive.to(device), val_negative.to(device)
            val_anchor_out, val_positive_out = model(val_anchor, val_positive)
            _, val_negative_out = model(val_anchor, val_negative)
            val_loss = criterion(val_anchor_out, val_positive_out, val_negative_out)
            val_running_loss += val_loss.item()

    # Calculate average validation loss for the epoch
    val_epoch_loss = val_running_loss / len(test_loader)
    validation_losses.append(val_epoch_loss)
    print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.4f}, Validation Loss: {val_epoch_loss:.4f}")

print("Finished Training Siamese Network")

# After training, save the Siamese Network model weights
torch.save(model.state_dict(), "/home/Student/s4647936/PatternAnalysis-2023/recognition/alzheimers_snn_s4647936/siamese_model.pth")
print("Saved Siamese Network model weights")

"""
Save and visualise results
"""
# Test to see number of images
# print(len(train_dataset)) # 17200
# print(len(test_dataset)) # 1820

# Save the loss curve
plt.figure()
plt.plot(training_losses, label='Training Loss')
plt.plot(validation_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Siamese Network Training vs Validation Loss')
plt.legend()
plt.savefig('siamese_train_vs_val_loss.png')

# Function to save images
def save_image(img, filename):
    # Select the first image from the batch
    img = img[0]

    # Move tensor to CPU and convert to numpy
    img = img.cpu().numpy()

    # Transpose from [channels, height, width] to [height, width, channels]
    img = img.transpose((1, 2, 0))
    
    # Convert to float and normalize if necessary
    if img.max() > 1:
        img = img.astype(float) / 255
        
    plt.figure()
    plt.imshow(img)
    plt.axis('off')  # Hide axes
    plt.savefig(filename)



# Save sample images after training
save_image(anchor, 'anchor_sample.png')
save_image(positive, 'positive_sample.png')
save_image(negative, 'negative_sample.png')

# --------- Visualize Embeddings using t-SNE ---------
# Extract embeddings and labels
all_embeddings = []
all_labels = []

# Assuming you have two classes: AD and NC. Let's assign them numeric labels.
# AD: 0, NC: 1
with torch.no_grad():
    for (anchor, _, _), label in train_loader:
        anchor = anchor.to(device)
        embedding, _ = model(anchor, anchor)
        all_embeddings.append(embedding.cpu().numpy())
        all_labels.extend(label.tolist())

print(f"Number of AD labels: {all_labels.count(0)}")
print(f"Number of NC labels: {all_labels.count(1)}")

all_embeddings = np.concatenate(all_embeddings)

# Reduce dimensionality using t-SNE
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(all_embeddings)

# Plot
plt.figure(figsize=(10, 7))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=all_labels, cmap='jet', alpha=0.5, edgecolors='w', s=40)
plt.colorbar()
plt.title('2D t-SNE of Embeddings')
plt.savefig('embeddings_tsne.png')

# --------- Extract Embeddings for the Entire Dataset ---------
train_embeddings = []
train_labels = []
test_embeddings = []
test_labels = []

with torch.no_grad():
    for anchor, _, _, label in train_loader:
        anchor = anchor.to(device)
        embedding, _ = model(anchor, anchor)
        train_embeddings.append(embedding.cpu().numpy())
        train_labels.extend(label.tolist())
    
    for anchor, _, _, label in test_loader:
        anchor = anchor.to(device)
        embedding, _ = model(anchor, anchor)
        test_embeddings.append(embedding.cpu().numpy())
        test_labels.extend(label.tolist())

train_embeddings = np.concatenate(train_embeddings)
test_embeddings = np.concatenate(test_embeddings)

# --------- Train Simple Classifier ---------
classifier = SimpleClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(classifier.parameters(), lr=0.001)

for epoch in range(num_epochs):
    # Train with embeddings
    for embeddings, labels in zip(train_embeddings, train_labels):
        optimizer.zero_grad()
        outputs = classifier(torch.tensor(embeddings).to(device))
        loss = criterion(outputs, torch.tensor(labels).to(device))
        loss.backward()
        optimizer.step()

# After training, save the classifier model weights
torch.save(classifier.state_dict(), "/home/Student/s4647936/PatternAnalysis-2023/recognition/alzheimers_snn_s4647936/classifier_model.pth")
print("Saved classifier model weights")

# --------- Evaluate Classifier ---------
correct = 0
total = 0
with torch.no_grad():
    for embeddings, labels in zip(test_embeddings, test_labels):
        outputs = classifier(torch.tensor(embeddings).to(device))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == torch.tensor(labels).to(device)).sum().item()

print(f"Accuracy of the classifier on test embeddings: {100 * correct / total}%")
