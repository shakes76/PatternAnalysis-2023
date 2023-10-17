import torch
import sys
import os
from dataset import *

# load dataset and trained model
os.chdir(sys.path[0])
model = torch.load('model.pth')
model.eval()
data = load_data(quiet=True, train_split=model.train_split, test_split=model.test_split)

# perform prediction on test set
embs, pred = model(data.X, data.edges) # forward pass
y_pred = pred[data.test_split].argmax(dim=1)

# print test accuracy
acc = model.accuracy(data.y[data.test_split], y_pred)
print('test accuracy:', round(acc.item(), 4))
pred = pred.argmax(dim=1)

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# produce a 2D embeddings plot via t-SNE
tsne = TSNE(random_state=1, n_iter=300, metric="cosine")
embs_pred = tsne.fit_transform(embs.detach().numpy())
embs_true = tsne.fit_transform(data.X)

# extract embeddings
X_pred = embs_pred[:, 0]
X_true = embs_true[:, 0]
Y_pred = embs_pred[:, 1]
Y_true = embs_true[:, 1]

# helper function to split embeddings by class label
def split_by_classification(X, Y):
    x1, x2, x3, x4 = [], [], [], []
    y1, y2, y3, y4 = [], [], [], []

    for i in range(len(X)):
        if pred[i] == 0:
            x1.append(X[i])
            y1.append(Y[i])
        elif pred[i] == 1:
            x2.append(X[i])
            y2.append(Y[i])
        elif pred[i] == 2:
            x3.append(X[i])
            y3.append(Y[i])
        else:
            x4.append(X[i])
            y4.append(X[i])

    return x1, x2, x3, x4, y1, y2, y3, y4

# seperate embeddings by classification
(x1_pred, x2_pred, x3_pred, x4_pred, 
 y1_pred, y2_pred, y3_pred, y4_pred) = split_by_classification(X_pred, Y_pred)
(x1_true, x2_true, x3_true, x4_true,
 y1_true, y2_true, y3_true, y4_true) = split_by_classification(X_true, Y_true)

# plot true class labels
fig, (ax0, ax1) = plt.subplots(1, 2)
ax0.scatter(x1_true, y1_true, alpha=.2, c='blue', label='0')
ax0.scatter(x2_true, y2_true, alpha=.2, c='red', label='1')
ax0.scatter(x3_true, y3_true, alpha=.2, c='green', label='2')
ax0.scatter(x4_true, y4_true, alpha=.2, c='pink', label='3')
ax0.set_title('True class labels')
ax0.set(xlabel='X', ylabel='Y')
ax0.legend(loc='lower right')

# plot predicted class labels
ax1.scatter(x1_pred, y1_pred, alpha=.2, c='blue', label='0')
ax1.scatter(x2_pred, y2_pred, alpha=.2, c='red', label='1')
ax1.scatter(x3_pred, y3_pred, alpha=.2, c='green', label='2')
ax1.scatter(x4_pred, y4_pred, alpha=.2, c='pink', label='3')
ax1.set_title('Predicted class labels')
ax1.set(xlabel='X', ylabel='Y')
ax1.legend(loc='lower right')

fig.suptitle('True and predicted class labels')
fig.tight_layout(pad=2)
plt.show()