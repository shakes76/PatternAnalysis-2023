import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.ops import nms, roi_align

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
        self.anchor_scales = [8, 16, 32]
        self.anchor_ratios = [0.5, 1, 2]

    def generate_base_anchors(self, feature_map_size):
        base_anchors = []
        for scale in self.anchor_scales:
            for ratio in self.anchor_ratios:
                w = scale * torch.sqrt(ratio)
                h = scale / torch.sqrt(ratio)
                x1, y1, x2, y2 = -w/2, -h/2, w/2, h/2
                base_anchors.append([x1, y1, x2, y2])
        base_anchors = torch.tensor(base_anchors)
        return base_anchors

    def apply_bbox_deltas(self, anchors, bbox_deltas):
        widths = anchors[:, 2] - anchors[:, 0]
        heights = anchors[:, 3] - anchors[:, 1]
        ctr_x = anchors[:, 0] + 0.5 * widths
        ctr_y = anchors[:, 1] + 0.5 * heights

        dx = bbox_deltas[:, 0::4]
        dy = bbox_deltas[:, 1::4]
        dw = bbox_deltas[:, 2::4]
        dh = bbox_deltas[:, 3::4]

        pred_ctr_x = ctr_x + dx * widths
        pred_ctr_y = ctr_y + dy * heights
        pred_w = torch.exp(dw) * widths
        pred_h = torch.exp(dh) * heights

        pred_boxes = torch.zeros_like(bbox_deltas)
        pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
        pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
        pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
        pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

        return pred_boxes

    def apply_nms(self, boxes, scores, threshold=0.5):
        keep_indices = nms(boxes, scores, threshold)
        return boxes[keep_indices], scores[keep_indices]

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        objectness = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        objectness_prob = F.softmax(objectness.view(objectness.size(0), 2, -1), dim=1)[:, 1, :]

        base_anchors = self.generate_base_anchors(x.shape[2:])
        anchors = base_anchors.repeat(x.shape[2] * x.shape[3], 1)

        refined_anchors = self.apply_bbox_deltas(anchors, bbox_deltas.squeeze())

        final_boxes, final_scores = self.apply_nms(refined_anchors, objectness_prob.squeeze())

        return final_boxes, final_scores


class RoIAlign(nn.Module):

    """
    The RoIAlign layer crops and resizes the feature maps from the backbone network for each object proposal.
    """

    def __init__(self, output_size, spatial_scale, sampling_ratio):
        super(RoIAlign, self).__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio

    def forward(self, input, boxes, box_indices):
        return roi_align(input, boxes, box_indices,
                         self.output_size, self.spatial_scale, self.sampling_ratio)


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
        self.roi_align = RoIAlign(output_size=(7, 7), spatial_scale=1 / 16, sampling_ratio=-1)
        self.classifier = Classifier(num_classes)
        self.box_regressor = BoxRegressor()
        self.mask_branch = MaskBranch()

        # Load the pretrained weights for the backbone network
        resnet = models.resnet50(weights=True)
        # Load the pretrained weights into the backbone network
        self.backbone.load_state_dict(resnet.state_dict(), strict=False)

    def forward(self, x, boxes, box_indices):
        x = self.backbone(x)
        proposals = self.rpn(x)
        rois = self.roi_align(x, boxes, box_indices)
        classification = self.classifier(rois)
        boxes = self.box_regressor(rois)
        masks = self.mask_branch(rois)
        return classification, boxes, masks



