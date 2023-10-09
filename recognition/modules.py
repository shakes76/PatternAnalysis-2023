import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.ops as ops
import torch.nn.functional as F
import ResNet

class RPN(nn.Module):

    """
    The RPN network takes as input the output features from the backbone network and outputs
    a set of rectangular object proposals, each with an objectness score.
    """

    def __init__(self, in_channels, mid_channels, n_anchor):

        """
        Initialize the RPN network.
        :param in_channels:  input channels
        :param mid_channels:  middle channels
        :param n_anchor:  number of anchors at each pixel
        """

        super(RPN, self).__init__()

        # Convolutional layer, use 3x3 kernel, 1 stride, and 1 padding to preserve the feature map size
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        self.relu1 = nn.ReLU(inplace=True)

        # Classification layer (objectiveness), 2 classes (object or not)
        self.cls_layer = nn.Conv2d(mid_channels, n_anchor * 2, 1, 1, 0)

        # Regression layer (bounding box coordinates)
        self.reg_layer = nn.Conv2d(mid_channels, n_anchor * 4, 1, 1, 0)

    def forward(self, x):

        """
        The forward function of the RPN network.
        :param x:  input feature map
        :return:  objectiveness score and bounding box coordinates
        """
        x = self.conv1(x)
        x = self.relu1(x)

        # Objectiveness score
        obj_score = self.cls_layer(x)

        # Bounding box coordinates
        bbox_deltas = self.reg_layer(x)

        return obj_score, bbox_deltas

class RoIAlign(nn.Module):

    """
    The RoIAlign layer crops and resizes the feature maps from the backbone network for each object proposal.
    """
    def __init__(self, output_size):
        super(RoIAlign, self).__init__()
        self.output_h, self.output_w = output_size

    def forward(self, features, rois):
        """
        :param features: feature maps [N, C, H, W]
        :param rois: rois [N, 5]  (batch_idx, x1, y1, x2, y2)
        :return: output [N, C, output_h, output_w]
        """

        n, c, h, w = features.size()
        outputs = []

        # Loop over each ROI
        for i, roi in enumerate(rois):
            batch_idx, x1, y1, x2, y2 = roi
            cur_feature = features[batch_idx]

            roi_h, roi_w = y2 - y1, x2 - x1
            bin_h, bin_w = roi_h / self.output_h, roi_w / self.output_w

            # Create sampling points and align with the input feature map
            sampling_points_x = torch.linspace(x1, x2, self.output_w + 1)[:-1] + bin_w / 2
            sampling_points_y = torch.linspace(y1, y2, self.output_h)[:-1] + bin_h / 2

            # Perform bilinear interpolation
            output_roi = F.grid_sample(
                cur_feature.unsqueeze(0).unsqueeze(0),
                torch.stack([
                    sampling_points_x.repeat(self.output_h),
                    sampling_points_y.repeat_interleave(self.output_w)
                ]).unsqueeze(0).unsqueeze(0),
                align_corners=False
            ).squeeze().permute(1, 2, 0)

            outputs.append(output_roi)

        return torch.stack(outputs)

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
        self.backbone = ResNet.ResNet(ResNet.Bottleneck, [3, 4, 6, 3])
        self.rpn = RPN()
        self.roi_align = RoIAlign()
        self.classifier = Classifier(num_classes)
        self.box_regressor = BoxRegressor()
        self.mask_branch = MaskBranch()

        # Load the pretrained weights for the backbone network
        resnet = models.resnet50(pretrained=True)
        # Load the pretrained weights into the backbone network
        self.backbone.load_state_dict(resnet.state_dict(), strict=False)

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


