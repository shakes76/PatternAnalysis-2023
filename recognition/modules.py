import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.ops as ops
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import ResNet50_Weights
from torchvision.ops import nms

import ResNet
from dataset import ISICDataset


class RPN(nn.Module):

    """
    The RPN network takes as input the output features from the backbone network and outputs
    a set of rectangular object proposals, each with an objectness score.
    """

    def __init__(self):
        super(RPN, self).__init__()
        self.conv = nn.Conv2d(2048, 512, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.cls_score = nn.Conv2d(512, 2 * 9, kernel_size=1, stride=1)
        self.bbox_pred = nn.Conv2d(512, 4 * 9, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        objectness = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)
        return objectness, bbox_deltas


class RoIAlign(nn.Module):

    """
    The RoIAlign layer crops and resizes the feature maps from the backbone network for each object proposal.
    """

    def __init__(self):
        super(RoIAlign, self).__init__()
        self.roi_align = ops.RoIAlign((7, 7), spatial_scale=1/16.0, sampling_ratio=-1)

    def forward(self, features, boxes, box_indices):
        return self.roi_align(features, boxes, box_indices)

class Classifier(nn.Module):

    """
    The classifier network takes as input the output features from the RoIAlign layer and outputs a classification
    """
    def __init__(self, num_classes):
        super(Classifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(2048 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class BoxRegressor(nn.Module):


    """
    The box regressor network takes as input the output features from the RoIAlign layer and outputs a set of bounding
    boxes.
    """
    def __init__(self):
        super(BoxRegressor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(2048 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class MaskBranch(nn.Module):

    """
    The mask branch network takes as input the output features from the RoIAlign layer and outputs a segmentation mask.
    """

    def __init__(self):
        super(MaskBranch, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2048, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, kernel_size=1, stride=1)
        )

    def forward(self, x):
        return self.conv(x)

class MaskRCNN(nn.Module):

    """
    The Mask R-CNN model structure.
    """
    def __init__(self, num_classes):

        """
        Initialize the Mask R-CNN model.
        :param num_classes: Number of classes in the dataset
        """

        super(MaskRCNN, self).__init__()
        self.backbone = ResNet.ResNet(ResNet.Bottleneck, [3, 4, 6, 3])
        self.rpn = RPN()
        self.roi_align = RoIAlign()
        self.classifier = Classifier(num_classes)
        self.box_regressor = BoxRegressor()
        self.mask_branch = MaskBranch()

        # Load the pretrained weights for the backbone network
        resnet = models.resnet50(weights=True)
        # Load the pretrained weights into the backbone network
        self.backbone.load_state_dict(resnet.state_dict(), strict=False)

    def forward(self, x):
        x = self.backbone(x)
        proposals = self.rpn(x)
        rois = self.roi_align(proposals)
        classification = self.classifier(rois)
        boxes = self.box_regressor(rois)
        masks = self.mask_branch(rois)
        return classification, boxes, masks


if __name__ == '__main__':
    train_dataset = ISICDataset(path="E:/comp3710/ISIC2018", type="Training")
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    # Initialize the model
    num_classes = 2  # For binary classification, we have 2 classes: 0 and 1
    model = MaskRCNN(num_classes)
    model = model.cuda()  # 如果使用GPU

    # 初始化优化器
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # 训练循环
    for epoch in range(10):  # 运行10个epoch，
        for i, (images, labels) in enumerate(train_loader):
            images = images.cuda()
            labels = labels.cuda()

            optimizer.zero_grad()

            classification, boxes, masks = model(images)

            break
