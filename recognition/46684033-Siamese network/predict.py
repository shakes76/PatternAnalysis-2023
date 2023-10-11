#predict.py
import torch
import dataset
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import random
import dataset
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning VUDA not Found. Using CPU")

train_path = r"C:/Users/wongm/Downloads/ADNI_AD_NC_2D/AD_NC/train"
test_path = r"C:/Users/wongm/Downloads/ADNI_AD_NC_2D/AD_NC/test"

model = torch.load(r"C:/Users/wongm/Desktop/COMP3710/project/siamese_epcoh60.pth")
model.eval()



transform = transforms.Compose([
    # transforms.Resize((128, 128)),
    transforms.ToTensor(),

])
train_path = r"C:/Users/wongm/Downloads/ADNI_AD_NC_2D/AD_NC/train"
test_path = r"C:/Users/wongm/Downloads/ADNI_AD_NC_2D/AD_NC/test"
trainset = torchvision.datasets.ImageFolder(root=train_path, transform=transform)
testset = torchvision.datasets.ImageFolder(root=test_path, transform=transform)
paired_testset = dataset.SiameseDatset_test(testset)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(paired_testset, batch_size=64,shuffle=True)
correct = 0
total = 0

with torch.no_grad():
    for i, (test_images,pos_images,neg_images,test_labels) in enumerate(test_loader):
        test_images = test_images.to(device)
        pos_images = pos_images.to(device)
        neg_images = neg_images.to(device)
        test_labels = test_labels.to(device)


        x1,y1 = model(test_images,pos_images)
        # accuracy test
        pos_distances = (y1 - x1).pow(2).sum(1)

        x2,y2 = model(test_images,neg_images)
        neg_distances = (y2 - x2).pow(2).sum(1)

        pred = torch.where(pos_distances < neg_distances, 1, 0)
        correct += (pred == 1).sum().item()
        total += test_labels.size(0)
        print(f"progress [{i}/{len(test_loader)}]")
    print(f"test accuracy {100*correct/total}%")