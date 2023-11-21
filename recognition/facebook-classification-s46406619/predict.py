import torch
import matplotlib.pyplot as plt
import sys
import os
from sklearn.manifold import TSNE
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
true = data.y[data.test_split]
acc = model.accuracy(true, y_pred)
print('test accuracy:', round(acc.item(), 4))
pred = pred.argmax(dim=1)

# produce a 2D embeddings plot via t-SNE
tsne = TSNE(random_state=1, n_iter=300, metric="cosine")
embs_full = tsne.fit_transform(embs.detach().numpy())
X = embs_full[:, 0]
Y = embs_full[:, 1]

# only the test set
embs_test = tsne.fit_transform(embs.detach().numpy()[data.test_split])
X_test = embs_test[:, 0]
Y_test = embs_test[:, 1]

# helper function to split embeddings by class label
def split_by_classification(X, Y, label):
    x1, x2, x3, x4 = [], [], [], []
    y1, y2, y3, y4 = [], [], [], []

    for i in range(len(X)):
        if label[i] == 0:
            x1.append(X[i])
            y1.append(Y[i])
        elif label[i] == 1:
            x2.append(X[i])
            y2.append(Y[i])
        elif label[i] == 2:
            x3.append(X[i])
            y3.append(Y[i])
        else:
            x4.append(X[i])
            y4.append(X[i])

    return x1, x2, x3, x4, y1, y2, y3, y4

# seperate embeddings by classification
(x1_full, x2_full, x3_full, x4_full, 
 y1_full, y2_full, y3_full, y4_full) = split_by_classification(X, Y, data.y)
(x1_true, x2_true, x3_true, x4_true,
 y1_true, y2_true, y3_true, y4_true) = split_by_classification(X_test, Y_test, true)
(x1_pred, x2_pred, x3_pred, x4_pred, 
 y1_pred, y2_pred, y3_pred, y4_pred) = split_by_classification(X_test, Y_test, y_pred)

# plot full embeddings
fig, (ax0, ax1, ax2) = plt.subplots(1, 3)
ax0.scatter(x1_full, y1_full, alpha=.2, c='blue', label='0')
ax0.scatter(x2_full, y2_full, alpha=.2, c='red', label='1')
ax0.scatter(x3_full, y3_full, alpha=.2, c='green', label='2')
ax0.scatter(x4_full, y4_full, alpha=.2, c='pink', label='3')
ax0.set_title('Full dataset embeddings')
ax0.set(xlabel='X', ylabel='Y')
ax0.legend(loc='lower right')

# plot true test set labels
ax1.scatter(x1_true, y1_true, alpha=.2, c='blue', label='0')
ax1.scatter(x2_true, y2_true, alpha=.2, c='red', label='1')
ax1.scatter(x3_true, y3_true, alpha=.2, c='green', label='2')
ax1.scatter(x4_true, y4_true, alpha=.2, c='pink', label='3')
ax1.set_title('True test set labels')
ax1.set(xlabel='X', ylabel='Y')
ax1.legend(loc='lower right')

# plot predicted test set labels
ax2.scatter(x1_pred, y1_pred, alpha=.2, c='blue', label='0')
ax2.scatter(x2_pred, y2_pred, alpha=.2, c='red', label='1')
ax2.scatter(x3_pred, y3_pred, alpha=.2, c='green', label='2')
ax2.scatter(x4_pred, y4_pred, alpha=.2, c='pink', label='3')
ax2.set_title('Predicted test set labels')
ax2.set(xlabel='X', ylabel='Y')
ax2.legend(loc='lower right')

fig.tight_layout(h_pad=.1)
plt.show()