import torch
import torchvision
from matplotlib.pyplot import imshow
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from modules import SiameseResNet
from train import test_dataloader, writer


# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set seeds for reproducibility
SEED = 1234
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

# Check if CUDA is available and set the device accordingly
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Initialize the number of correct predictions to zero
correct = 0
total = 0

# Load the trained Siamese network model
net = SiameseResNet().to(device)
net.load_state_dict(torch.load("model.pth", map_location=device))

# Set the network to evaluation mode
net.eval()

# Define a threshold for classification based on the distance between outputs
threshold = 0.5

# Create lists to store the true labels and predicted labels
labels_list = []
predict_list = []

# Loop through the test data and make predictions using the trained model
for i, data in enumerate(test_dataloader, 0):
    # Get the images and labels from the data
    img0, img1, label = data

    # Move the images and labels to the appropriate device
    img0, img1, label = img0.to(device), img1.to(device), label.to(device)

    # Get the outputs from the Siamese network
    output1, output2 = net(img0, img1)

    # Calculate the Euclidean distance between the two outputs
    distance = torch.nn.functional.pairwise_distance(output1, output2)

    # Classify the pair based on the distance
    pred = (distance < threshold).float()

    # Update the total number of predictions
    total += label.size(0)

    # Update the number of correct predictions
    correct += (pred == label).sum().item()

    # Concatenate and display the images
    concatenated = torch.cat((img0, img1), 0)
    imshow(torchvision.utils.make_grid(concatenated.cpu().detach()))

    # Print the predicted and actual labels
    print(f"Predicted: {pred.item()}, Actual: {label.item()}")

    labels_list.append(label.item())
    predict_list.append(pred.item())


# Calculate the accuracy
accuracy = 100 * correct / total
print(f"Accuracy: {accuracy}%")

# Plot the confusion matrix
cm = confusion_matrix(labels_list, predict_list)
print(cm)
plt.figure(figsize=(10, 10))
plt.title('Confusion Matrix')
sns.heatmap(cm, annot=True, fmt='g')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.show()

# Log the accuracy to tensorboard
writer.add_scalar('Test Accuracy', accuracy)

# Close the tensorboard writer
writer.close()
