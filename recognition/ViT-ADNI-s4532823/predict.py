"""
Load saved model and show a sample of predictions, alongside the true labels.
"""
import torch
from dataset import ADNIDataset, ADNI_PATH, TEST_TRANSFORM
import random
from torchvision import transforms


def do_prediction(model, device: torch.device, num_predictions: int = 1):
    test_dataset = ADNIDataset(ADNI_PATH, train=False, transform=TEST_TRANSFORM)
    for i in range(num_predictions):
        # load random image & label
        idx = random.randint(0,test_dataset.__len__()-1)
        image, label = test_dataset.__getitem__(idx)
        # apply necessary transforms
        image = image.unsqueeze(0).to(device)
        image = transforms.CenterCrop(224)(image)
        image = transforms.Grayscale(3)(image)
        # make prediction
        output = model(image)
        _, predicted = torch.max(output, 1)
        pred = int(predicted[0].item())
        label_strs = {0: "Cognitive normal", 1: "Alzheimer's disease"}
        print(f"Predicted {label_strs[pred]}, actually {label_strs[label]}")
        # show image used for prediction
        image = image.squeeze(0)
        image = transforms.ToPILImage()(image)
        image.show()
        

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning CUDA not found. Using CPU")

# Load model
model = torch.load("adni_vit.pt").to(device)

do_prediction(model, device)