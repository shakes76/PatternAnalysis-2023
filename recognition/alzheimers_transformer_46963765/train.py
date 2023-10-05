import dataset
from modules import *

#put in training loop

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ADNI_Transformer(depth=3)
model.to(device=device)

dataset = ds.ADNI_Dataset()
train_loader = dataset.get_train_loader()
test_loader = dataset.get_test_loader()


model.train()
for j, (images, labels) in  enumerate(test_loader):
    if images.size(0) == 32:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
