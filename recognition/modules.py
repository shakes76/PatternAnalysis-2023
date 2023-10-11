# modules.py
import torchvision.models as models


class MaskRCNNModule:
    def __init__(self, pretrained=True):
        self.model = models.detection.maskrcnn_resnet50_fpn(pretrained=pretrained)

    def get_model(self):
        return self.model
