import torch
import torchvision
from torch import nn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


def get_model_instance_segmentation(num_classes):
    # 加载预训练的Mask R-CNN模型
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # 获取分类器的输入特征数
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # 重新定义模型的分类器部分
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # 获取掩膜分类器的输入特征数
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels

    # 定义新的掩膜预测器
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)

    return model

class ImageClassifier(torch.nn.Module):
    def __init__(self, backbone, num_classes: int):
        super().__init__()
        self.backbone = backbone
        input_dim = 256
        hidden_dim=48
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        # 第二个全连接层，接ReLU激活函数和BatchNorm
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        # 输出层，无激活函数
        self.fc_out = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        with torch.no_grad():
            x = self.backbone(x)['0']  # Assuming we take the output of the last layer (tuple)
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc_out(x)
        return x