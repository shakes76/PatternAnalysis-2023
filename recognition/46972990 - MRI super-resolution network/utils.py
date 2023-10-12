import torch
from torchviz import make_dot
from modules import ESPCN

MODEL_PATH = "recognition\\46972990 - MRI super-resolution network\\model.pth"
IMG_WIDTH = 256
IMG_HEIGHT = 240

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
model = ESPCN(upscale_factor=4, channels=1)
model.load_state_dict(torch.load(MODEL_PATH))
model = model.to(device).eval()

# Dummy input for the forward pass
x = torch.randn(1, 1, IMG_HEIGHT, IMG_WIDTH).to(device)
y = model(x)

dot = make_dot(y, params=dict(model.named_parameters()))
dot.view()