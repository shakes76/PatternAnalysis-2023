from torchvision.datasets.mnist import MNIST
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

transform = ToTensor()
train_set = MNIST(root='./../datasets', train=True, download=True, transform=transform)

train_loader = DataLoader(train_set, shuffle=True, batch_size=1)
for batch in train_loader:
    x, y = batch
    print("x: ", x[0][0][13])
    print("y: ", y)
    print(x.shape)
    print(y.shape)
