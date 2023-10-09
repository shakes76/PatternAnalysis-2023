import torch
import torch.nn as nn
import torchvision.models as models

class Backbone(nn.Module):

    """
    The backbone network is responsible for extracting features from the input image.
    """
    def __init__(self, pretrained=True):

        """
        Initialize the backbone network.
        :param pretrained: whether to use pretrained weights from ImageNet
        """

        super(Backbone, self).__init__()
        resnet = models.resnet50(pretrained=pretrained)
        self.features = nn.Sequential(*list(resnet.children())[:-2])

    def forward(self, x):

        x = self.features(x)
        return x

class RPN(nn.Module):

    """
    The RPN network takes as input the output features from the backbone network and outputs
    a set of rectangular object proposals, each with an objectness score.
    """
    def __init__(self):
        super(RPN, self).__init__()


    def forward(self, x):

        return x

class RoIAlign(nn.Module):

    """
    The RoIAlign layer crops and resizes the feature maps from the backbone network for each object proposal.
    """
    def __init__(self):
        super(RoIAlign, self).__init__()


    def forward(self, x):

        return x

class Classifier(nn.Module):

    """
    The classifier network takes as input the output features from the RoIAlign layer and outputs a classification
    """
    def __init__(self, num_classes):
        super(Classifier, self).__init__()


    def forward(self, x):

        return x

class BoxRegressor(nn.Module):


    """
    The box regressor network takes as input the output features from the RoIAlign layer and outputs a set of bounding
    boxes.
    """
    def __init__(self):
        super(BoxRegressor, self).__init__()


    def forward(self, x):

        return x


class MaskBranch(nn.Module):

    """
    The mask branch network takes as input the output features from the RoIAlign layer and outputs a segmentation mask.
    """
    def __init__(self):
        super(MaskBranch, self).__init__()


    def forward(self, x):

        return x

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
        self.backbone = Backbone()
        self.rpn = RPN()
        self.roi_align = RoIAlign()
        self.classifier = Classifier(num_classes)
        self.box_regressor = BoxRegressor()
        self.mask_branch = MaskBranch()

    def forward(self, x):

        """
        The forward function of the Mask R-CNN model.
        :param x:  input image
        :return:  classification, bounding box, and mask outputs
        """

        x = self.backbone(x)
        proposals = self.rpn(x)
        rois = self.roi_align(proposals)
        classification = self.classifier(rois)
        boxes = self.box_regressor(rois)
        masks = self.mask_branch(rois)
        return classification, boxes, masks
