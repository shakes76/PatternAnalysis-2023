import torch
from dataset import *
from modules import *

model = torch.load('C:/Area-51/2023-sem2/COMP3710/PatternAnalysis-2023/recognition/facebook-classification-s46406619/model.pth')
model.eval()

data = load_data(quiet=True, train_split=model.train_split, test_split=model.test_split)

# perform prediction on test set
embeddings, pred = model(data.X, data.edges) # forward pass

# print test accuracy
y_test = data.y[data.test_split]
y_pred = pred[data.test_split].argmax(dim=1)
acc = model.accuracy(y_test, y_pred)
print('test accuracy:', round(acc.item(), 4))
pred = pred.argmax(dim=1)

# t-SNE embeddings plot
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# calculate 2D embeddings
tsne = TSNE(random_state=1, n_iter=300, metric="cosine")
embs = tsne.fit_transform(embeddings.detach().numpy())
X = embs[:, 0]
Y = embs[:, 1]

# seperate embeddings by classification
X_one = []
Y_one = []
X_two = []
Y_two = []
X_three = []
Y_three = []
X_four = []
Y_four = []
for i in range(len(X)):
    if pred[i] == 0:
        X_one.append(X[i])
        Y_one.append(Y[i])
    elif pred[i] == 1:
        X_two.append(X[i])
        Y_two.append(Y[i])
    elif pred[i] == 2:
        X_three.append(X[i])
        Y_three.append(Y[i])
    else:
        X_four.append(X[i])
        Y_four.append(X[i])

# plot embeddings
fig, ax = plt.subplots()
ax.scatter(X_one, Y_one, alpha=.1, c='blue')
ax.scatter(X_two, Y_two, alpha=.1, c='red')
ax.scatter(X_three, Y_three, alpha=.1, c='green')
ax.scatter(X_four, Y_four, alpha=.1, c='pink')
plt.show()