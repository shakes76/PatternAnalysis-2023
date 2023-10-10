import torch
from dataset import *
from modules import *

model = torch.load('C:/Area-51/2023-sem2/COMP3710/PatternAnalysis-2023/recognition/facebook-classification-s46406619/model.pth')
model.eval()

data = load_data(quiet=True, train_split=model.train_split, test_split=model.test_split)

# perform prediction on test set
h, z = model(data.X, data.edges) # forward pass

# keep only testing elements
y_pred = data.y[data.test_split]
z = z[data.test_split]

acc = model.accuracy(y_pred, z.argmax(dim=1)) # calculate accuracy
print('test accuracy:', acc.item())