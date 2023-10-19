import torch
from modules import ViT
from dataset import get_dataloaders

def predict():
    batch_size = 64
    workers = 4
    image_size = 224
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ViT(num_channels=3, embed_dim=768, patch_size=16).to(device)
    model.load_state_dict(torch.load('visual_transformer'))
    _, test_loader = get_dataloaders(batch_size, workers, image_size)
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
    predict()