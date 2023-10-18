# containing the source code of the components of your model.
# Each component must be implementated as a class or a function

import torch
import torch.nn as nn
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
device = torch.device('cuda')

# Build CNN network and get its embedding vector
class Embedding(nn.Module):
    def __init__(self):
        super(Embedding, self).__init__()
        self.conv = nn.Sequential(

            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1), # size: 256*240 -> 128*120

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1), # size: 128*120 -> 64*60

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1), # size: 64*60 -> 32*30
            )

        self.fc = nn.Sequential(
            nn.Linear(64*32*30, 512),
            nn.ReLU(inplace=True),

            nn.Linear(512, 256),
            )

    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size()[0], -1)
        out = self.fc(out)
        return out


# construct the triplet loss
# formular: L = (1 - y) * 1/2 * D^2 + y * 1/2 * max(0, m - D)^2 
# where D = sample distance, m = margin, y = label, same: label = 0; diff, label = 1
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.5):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, img1, img2, label):
        # calculate euclidean distance  
        distance = (img1 - img2).pow(2).sum(1).sqrt()

        # calculate loss, use relu to ensure loss are non-negative
        loss_same = (1 - label) * 0.5 * (distance ** 2)
        loss_diff = label * 0.5 * torch.relu(self.margin - distance).pow(2)
        loss = loss_same + loss_diff

        return loss.mean()


# get the trained embedding network
def extract_embeddings(loader, model):
    model.eval()
    embeddings = []
    labels_list = []

    with torch.no_grad():
        for img1, img2, labels in loader:
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
            
            emb1 = model.get_embedding(img1)
            emb2 = model.get_embedding(img2)
            
            embeddings.append(emb1.cpu())
            embeddings.append(emb2.cpu())
            
            labels_list.extend(labels.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())

    embeddings = torch.cat(embeddings, dim=0)
    return embeddings, labels_list


# construct the siamese network
class SiameseNet(nn.Module):
    def __init__(self, embedding):
        super(SiameseNet, self).__init__()
        self.embedding = embedding

    def forward(self, img1, img2):
        emb1 = self.embedding(img1)
        emb2 = self.embedding(img2)
        return emb1, emb2

    def get_embedding(self, x):
        return self.embedding(x)


# use embedding net to train knn clasifier
def knn(train_loader, val_loader, model, n_neighbors=5):
    
    # Extract embeddings from the train set
    train_embeddings, train_labels = extract_embeddings(train_loader, model)
    
    # Train a KNN classifier
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(train_embeddings, train_labels)
    
    # Extract embeddings from the validation set
    val_embeddings, val_labels = extract_embeddings(val_loader, model)
    
    # Predict the labels of the validation set
    test_preds = knn.predict(val_embeddings)
    
    # Calculate the accuracy
    accuracy = accuracy_score(val_labels, test_preds)
    print(f"KNN Accuracy: {accuracy:.4f}")
    with open ("D:/Study/GitHubDTClone/COMP3710A3/PatternAnalysis-2023/recognition/s4627382_SiameseNetwork/knn.pkl", "wb") as f:
        pickle.dump(knn, f)
    return accuracy