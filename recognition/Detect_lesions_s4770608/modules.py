import torch
import torchvision
from torch import nn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import segmentation_models_pytorch as smp

def get_model_instance_segmentation(num_classes):
    # 加载预训练的Mask R-CNN模型
    model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(pretrained=True)

    # 获取分类器的输入特征数
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # 重新定义模型的分类器部分
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # 获取掩膜分类器的输入特征数
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels

    # 定义新的掩膜预测器
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)

    for name, para in model.named_parameters():
        para.requires_grad =True
    return model
def get_deeplab_model(num_classes):
    model = smp.DeepLabV3Plus(
        encoder_name="efficientnet-b4",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=2,  # model output channels (number of classes in your dataset)
        aux_params={'classes':3}
    )
    return  model
class ImageClassifier(torch.nn.Module):
    def __init__(self, backbone, num_classes: int):
        super().__init__()
        self.backbone = backbone
        input_dim = 2048
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):

        x = self.backbone(x)['3']  # Assuming we take the output of the last layer (tuple)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x