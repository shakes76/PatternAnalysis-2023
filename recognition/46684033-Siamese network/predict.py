#predict.py
import torch
import dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning VUDA not Found. Using CPU")

train_path = r"C:/Users/wongm/Downloads/ADNI_AD_NC_2D/AD_NC/train"
test_path = r"C:/Users/wongm/Downloads/ADNI_AD_NC_2D/AD_NC/test"

model = torch.load(r"C:/Users/wongm/Desktop/COMP3710/project/siamesev2.pth")
model.eval()

train_loader, test_loader = dataset.load_data2(train_path, test_path)

correct = 0
total = 0

for trial in range(5):
    with torch.no_grad():
        for i, ((images1,images2), labels) in enumerate(test_loader):
            images1 = images1.to(device)
            images2 = images2.to(device)
            labels = labels.to(device)
            output = model(images1,images2).squeeze()


            pred = torch.where(output > 0.5, 1, 0)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

            print(f"test accuracy {100*correct/total}")