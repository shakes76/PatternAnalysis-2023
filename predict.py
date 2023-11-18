import torch
from modules import ViT
from dataset import get_dataloaders

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning CUDA not found. Using CPU")

def predict(model_dict: str):
    batch_size = 64
    workers = 4
    image_size = 224
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ViT().to(device)
    model.load_state_dict(torch.load(model_dict))
    _, test_loader, _ = get_dataloaders(batch_size, workers, image_size, dataroot="AD_NC", rgb=False)
    model.eval()

    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        print('Test Accuracy: {} %'.format(100 * correct / total))

if __name__ == '__main__':
    predict('visual_transformer')