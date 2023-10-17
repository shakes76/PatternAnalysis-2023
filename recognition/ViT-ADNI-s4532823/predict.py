"""
Load saved model and show a sample of predictions, alongside the true labels.
"""
import torch
from dataset import ADNIDataset, ADNI_PATH, TEST_TRANSFORM
import random

def do_prediction(model, num_predictions: int = 1):
    test_dataset = ADNIDataset(ADNI_PATH, train=False, transform=TEST_TRANSFORM)
    for i in range(num_predictions):
        # load random image & label
        idx = random.randint(0, test_dataset.__len__()-1)
        image, label = test_dataset.__getitem__(idx)
        # apply necessary transforms
        # make prediction
        output = model(image)
        _, predicted = torch.max(output, 1)
        pred = predicted[0]

        # show prediction vs real result
        image.show()
        print(f"Predicted {pred}, actually {label}")

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning CUDA not found. Using CPU")

# Load model
model = torch.load("adni_vit.pt").to(device)

do_prediction(model)