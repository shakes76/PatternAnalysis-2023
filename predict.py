import torch
from train import test
#from dataset import trainloader, valloader, testloader
from dataset import returnDataLoaders


import matplotlib.pyplot as plt

model = torch.jit.load('model_trained.pt')
model.eval()

epoch = list(range(75))
accuracies = [
    0.4939, 0.6016, 0.5729, 0.5243, 0.5050, 0.5343, 0.5786, 0.6042, 0.5075, 0.5056,
    0.5925, 0.5694, 0.5604, 0.5567, 0.6067, 0.5681, 0.5337, 0.5422, 0.5243, 0.5431,
    0.5194, 0.5806, 0.5905, 0.5794, 0.5880, 0.6260, 0.5737, 0.5675, 0.5709, 0.5942,
    0.6061, 0.6022, 0.5829, 0.6149, 0.6022, 0.5905, 0.5786, 0.5704, 0.6399, 0.6260,
    0.5843, 0.5856, 0.5999, 0.6118, 0.5885, 0.5999, 0.6104, 0.5536, 0.6130, 0.5612,
    0.6374, 0.6106, 0.6155, 0.6388, 0.5968, 0.6167, 0.6167, 0.5587, 0.5874, 0.5601,
    0.6019, 0.5905, 0.6093, 0.6013, 0.6005, 0.5987, 0.5997, 0.5944, 0.6024, 0.6036,
    0.6079, 0.6036, 0.6013, 0.6050, 0.6013, 0.6812
]

plt.figure(figsize=(12, 6))
plt.plot(epoch, accuracies, marker='o', linestyle='-', color='b', label='Accuracy')
plt.title('Accuracy vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()
