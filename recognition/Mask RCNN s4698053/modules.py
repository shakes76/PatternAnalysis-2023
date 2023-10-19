from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
import torch.nn as nn
from math import log

# https://pytorch.org/vision/main/models/generated/torchvision.models.detection.maskrcnn_resnet50_fpn_v2.html
# Mask RCNN with a ResNet-50-FPN backbone

def getModel(num_classes=3) -> maskrcnn_resnet50_fpn_v2:
    model = maskrcnn_resnet50_fpn_v2(num_classes=num_classes)

    return model
