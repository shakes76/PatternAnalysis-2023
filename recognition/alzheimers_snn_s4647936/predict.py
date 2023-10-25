import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from modules import SiameseNetwork, SimpleClassifier
from torchvision import transforms
from dataset import TripletDataset
import seaborn as sns
from sklearn.metrics import confusion_matrix
import datetime

# Generate a unique filename with a timestamp
current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

def get_unique_filename(base_filename):
    return f"{base_filename}_{current_time}.png"

def plot_confusion_matrix(y_true, y_pred, classes, base_filename):
    output_filename = get_unique_filename(base_filename)
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

# Dataset instance for testing
test_dataset = TripletDataset(root_dir="/home/Student/s4647936/PatternAnalysis-2023/recognition/alzheimers_snn_s4647936/AD_NC", mode='test', transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)

# GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained Siamese Network
siamese_model = SiameseNetwork().to(device)
siamese_model.load_state_dict(torch.load("/home/Student/s4647936/PatternAnalysis-2023/recognition/alzheimers_snn_s4647936/siamese_model.pth", map_location=device))
siamese_model.eval()

# Load the trained Simple Classifier
classifier = SimpleClassifier().to(device)
classifier.load_state_dict(torch.load("/home/Student/s4647936/PatternAnalysis-2023/recognition/alzheimers_snn_s4647936/classifier_model.pth", map_location=device))
classifier.eval()

# Extract embeddings and labels for testing
test_embeddings = []
test_labels = []

with torch.no_grad():
    for anchor, _, _, label in test_loader:
        anchor = anchor.to(device)
        embedding, _ = siamese_model(anchor, anchor)
        test_embeddings.append(embedding.cpu().numpy())
        test_labels.extend(label.tolist())

test_embeddings = np.concatenate(test_embeddings)

# Visualise the embeddings using t-SNE
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(test_embeddings)

plt.figure(figsize=(10, 7))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=test_labels, cmap='jet', alpha=0.5, edgecolors='w', s=40)
plt.colorbar()
plt.title('2D t-SNE of Test Embeddings')
plt.savefig(get_unique_filename('test_embeddings_tsne'))

# Evaluate classifier on embeddings
test_embeddings_tensor = torch.tensor(test_embeddings).float().to(device)
test_labels_tensor = torch.tensor(test_labels).to(device)

outputs = classifier(test_embeddings_tensor)
_, predicted = torch.max(outputs, 1)

# Plot confusion matrix for the classifier
class_names = ["AD", "NC"]
plot_confusion_matrix(test_labels, predicted.cpu().numpy(), class_names, "test_classifier_confusion_matrix")

correct = (predicted == test_labels_tensor).sum().item()
total = test_labels_tensor.size(0)

print(f"Accuracy of the classifier on test embeddings: {100 * correct / total}%")
