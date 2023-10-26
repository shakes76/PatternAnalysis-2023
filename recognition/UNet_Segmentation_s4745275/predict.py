# showing example usage of your trained model. Print out any results and / or provide visualisations where applicable

from modules import UNet
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(in_channels=6, num_classes=2)  # Adjust parameters if they've changed
model.load_state_dict(torch.load("recognition/Problem_47452752/model.pth"))
model = model.to(device)