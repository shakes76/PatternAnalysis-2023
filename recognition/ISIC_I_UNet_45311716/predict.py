import torch
import torchvision
from modules import ImprovedUNet
from dataset import UNetData

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning CUDA not Found. Using CPU")

# Path to image directory
data_path = 'C:/Users/mlamt/OneDrive/UNI/2023/Semester 2/COMP3710/Code/data/ISIC2018/' 
# Path to trained model
model_path = 'C:/Users/mlamt/OneDrive/UNI/2023/Semester 2/COMP3710/Code/data/ImprovedUNet.pt'
# Location to save predicted and actual masks
image_path = 'C:/Users/mlamt/OneDrive/UNI/2023/Semester 2/COMP3710/Code/data/predict_images/'

# Hyper-parameters
image_height = 512 
image_width = 512
batch_size = 8

# Following function is from github:
# Reference: https://github.com/pytorch/pytorch/issues/1249#issuecomment-305088398
def dice_loss(input, target):
    smooth = 1.

    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()

    return 1 - ((2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))

# Load trained model
try:
    model = torch.load(model_path)
except:
    print("Model couldn't be found!")
    model = ImprovedUNet(in_channels=3, out_channels=1).to(device)

# Get test data loader
test_data = UNetData(path=data_path, height=image_height, width=image_width, batch=batch_size).get_test_loader()

print(' - - Start Predictions - - ')
model.eval()
with torch.no_grad():
    for i, (image, mask) in enumerate(test_data):
        image = image.to(device)
        mask = mask.to(device)

        pred = (model(image) > 0.5).float()
        mask = (mask.float())

        dice_score = 1 - dice_loss(pred, mask)

        print(f"Dice Score of Prediction {i}: " + "{:.4f}".format(dice_score))

        torchvision.utils.save_image(pred, f"{image_path}Prediction_batch_{i}.png")
        torchvision.utils.save_image(mask, f"{image_path}Actual_batch_{i}.png")

        if i > 0:
            break
