import dataset
from modules import *

#put in training loop

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ADNI_Transformer(depth=3)
model.to(device=device)

dataset = ds.ADNI_Dataset()
train_loader = dataset.get_train_loader()

model.train()
for j, (images, labels) in  enumerate(train_loader):
    images = images.to(device)
    labels = labels.to(device)
    outputs = model(images)
    print(outputs.shape)
