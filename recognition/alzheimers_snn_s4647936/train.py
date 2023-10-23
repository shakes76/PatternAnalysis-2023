import os
import torch
import matplotlib.pyplot as plt
from dataset import TripletDataset 
from modules import SiameseNetwork, TripletLoss
from torchvision import transforms
import torch.optim as optim
from sklearn.decomposition import PCA
import numpy as np

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

# Training loop
losses = []

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

    epoch_loss = running_loss / len(train_loader)
    losses.append(epoch_loss)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

print("Finished Training")

# Save the model's parameters
torch.save(model.state_dict(), "siamese_model.pth")

# --------- Begin Validation/Testing ---------
model.eval()  # set the model to evaluation mode
correct = 0
total = 0

# DataLoader setup for test dataset
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

with torch.no_grad():  # deactivate autograd engine to reduce memory usage and speed up computations
    for batch_idx, (anchor, positive, negative) in enumerate(test_loader):
        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
        
        # Forward pass
        anchor_out, positive_out = model(anchor, positive)
        _, negative_out = model(anchor, negative)

        # Compute triplet loss
        loss = criterion(anchor_out, positive_out, negative_out)

        # You might want to add some logic here to determine "correctness", depending on how you define it for your problem
        # For instance, if the distance between anchor and positive is less than between anchor and negative, consider it "correct"
        positive_distance = torch.dist(anchor_out, positive_out)
        negative_distance = torch.dist(anchor_out, negative_out)

        if positive_distance < negative_distance:
            correct += 1
        total += 1

print(f"Accuracy on test set: {100 * correct / total}%")
# --------- End Validation/Testing ---------

"""
Save and visualise results
"""
# Test to see number of images
# print(len(train_dataset)) # 17200
# print(len(test_dataset)) # 1820

# Save the loss curve
plt.figure()
plt.plot(losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.savefig('loss_curve.png')

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


# --------- Visualize Embeddings ---------
# Extract embeddings and labels
all_embeddings = []
all_labels = []

# Assuming you have two classes: AD and NC. Let's assign them numeric labels.
# AD: 0, NC: 1
with torch.no_grad():
    for idx, (anchor, _, _) in enumerate(loader):
        anchor = anchor.to(device)
        embedding, _ = model(anchor, anchor)
        all_embeddings.append(embedding.cpu().numpy())
        
        # Assign labels based on paths. You can adjust this based on your dataset setup.
        if idx < len(train_dataset.ad_paths):
            all_labels.extend([0] * anchor.size(0))  # AD
        else:
            all_labels.extend([1] * anchor.size(0))  # NC

all_embeddings = np.concatenate(all_embeddings)

# Reduce dimensionality using PCA
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(all_embeddings)

# Plot
plt.figure()
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=all_labels, cmap='jet', alpha=0.5)
plt.colorbar()
plt.title('2D PCA of Embeddings')
plt.savefig('embeddings_pca.png')
