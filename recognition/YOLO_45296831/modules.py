import torch
from ultralytics import YOLO

def get_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        print("Warning CUDA not Found. Using CPU")
    return 'cpu'

def get_yolo(device, pretrained=True):
    if pretrained:
        model = YOLO("yolov8n.pt")
    else:
        model = YOLO("yolov8n.yaml")
    model.to(device)
    return model

def detect_image(img, model):
    model.predict(img)