import torch
import torch.nn as nn
from torchvision import transforms
from modules import SiameseNetwork
from dataset import load_classify_data
from sklearn.metrics import accuracy_score

snn_path = 'SNN.pth'
siamese_network = SiameseNetwork()
siamese_network.load_state_dict(torch.load(snn_path,map_location=torch.device('cpu')))
siamese_network.eval()

class Classifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Hyperparameters
input_size = 2  
hidden_size = 128
num_classes = 2  
batch_size = 32

classify_data_loader = load_classify_data(testing=True, batch_size=batch_size)

classifier = Classifier(input_size, hidden_size, num_classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classifier.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-4)

num_epochs = 45  
for epoch in range(num_epochs):
    classifier.train()
    for images, _, labels in classify_data_loader:
        optimizer.zero_grad()
        images = images.to(device)
        labels = labels.long().to(device)
        
        with torch.no_grad():
            embeddings = siamese_network.forward_one(images)

        outputs = classifier(embeddings)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

classifier.eval()
test_loss = 0
all_preds = []
all_labels = []

with torch.no_grad():
    for images, _, labels in classify_data_loader:
        images = images.to(device)
        labels = labels.to(device)

        embeddings = siamese_network.forward_one(images)

        outputs = classifier(embeddings)

        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = accuracy_score(all_labels, all_preds)
print(f'Test Accuracy: {accuracy:.2f}')
