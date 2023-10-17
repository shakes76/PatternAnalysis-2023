import torch
from ultralytics import YOLO

def get_device():
    '''
    Obtains the device that will be used by pytorch and yolo to train the model

    returns: the device that will be used
    '''
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if not torch.cuda.is_available():
        print("Warning CUDA not Found. Using CPU")
    return device



def get_yolo(device, pretrained=True):
    '''
    Defines which yolo model to use, then sends it to the specified device for the model
    
    device: device to be trained on
    pretrained: specifies whether to use default model or custom model
    returns: model'''
    if pretrained:
        model = YOLO("yolov8n.pt")
    else:
        model = YOLO("yolov8n.yaml")
    model.to(device)
    return model
