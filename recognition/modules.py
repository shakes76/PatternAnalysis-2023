import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.ops import nms, roi_align

import ResNet
from dataset import ISICDataset

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import nms


class RPN(nn.Module):
    def __init__(self, in_channels=2048, anchor_sizes=[128, 256, 512], anchor_ratios=[0.5, 1, 2]):
        super(RPN, self).__init__()

        self.conv = nn.Conv2d(in_channels, 512, 3, 1, 1)
        self.cls_score = nn.Conv2d(512, len(anchor_sizes) * len(anchor_ratios) * 2, 1, 1, 0)
        self.bbox_pred = nn.Conv2d(512, len(anchor_sizes) * len(anchor_ratios) * 4, 1, 1, 0)

        self.anchor_sizes = anchor_sizes
        self.anchor_ratios = anchor_ratios

    def forward(self, x):
        # RPN forward to get objectness and bbox_deltas
        x = self.conv(x)
        objectness_score = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        # Generate anchors based on the feature map dimensions
        feature_map_size = (x.size(2), x.size(3))
        anchors = self.generate_anchors(feature_map_size)

        # Apply bbox_deltas to anchors
        refined_anchors = self.apply_deltas_to_anchors(anchors, bbox_deltas.detach())

        # Filter anchors using NMS and objectness score
        nms_indices = self.filter_anchors(refined_anchors, objectness_score)
        final_anchors = refined_anchors[nms_indices]

        return final_anchors

    def generate_anchors(self, feature_map_size, stride=32):
        anchors = []
        for y in range(feature_map_size[0]):
            for x in range(feature_map_size[1]):
                for size in self.anchor_sizes:
                    for ratio in self.anchor_ratios:
                        center_x, center_y = stride * (x + 0.5), stride * (y + 0.5)
                        height, width = size * torch.sqrt(torch.tensor(1.0 / ratio)), size * torch.sqrt(
                            torch.tensor(ratio))
                        anchors.append([center_x - 0.5 * width, center_y - 0.5 * height, center_x + 0.5 * width,
                                        center_y + 0.5 * height])

        anchors = torch.tensor(anchors, dtype=torch.float32)
        print(f"anchors shape when generate: {anchors.shape}")

        return anchors

    def apply_deltas_to_anchors(self, anchors, deltas):

        """
        Applies the deltas to the anchors to get the refined anchors.
        :param anchors:  [9, 4], containing the coordinates of the anchors
        :param deltas:  [4, 9, 8, 8], containing the deltas to be applied to the anchors
        :return: [4, 9, 8, 8], containing the refined anchors
        """

        print(f"anchors shape before apply deltas: {anchors.shape}")

        # Expand dims to apply broadcasting
        expanded_anchors = anchors.view(1, 9, 8, 8, 4).expand(4, 9, 8, 8, 4).to(deltas.device)

        # Expand dims to apply broadcasting
        expanded_x = expanded_anchors[..., 0]
        expanded_y = expanded_anchors[..., 1]
        expanded_w = expanded_anchors[..., 2] - expanded_anchors[..., 0]
        expanded_h = expanded_anchors[..., 3] - expanded_anchors[..., 1]

        # Extract deltas
        dx = deltas[:, 0::4, :, :]
        dy = deltas[:, 1::4, :, :]
        dw = deltas[:, 2::4, :, :]
        dh = deltas[:, 3::4, :, :]

        # Apply deltas
        pred_ctr_x = expanded_x + dx * expanded_w
        pred_ctr_y = expanded_y + dy * expanded_h
        pred_w = torch.exp(dw) * expanded_w
        pred_h = torch.exp(dh) * expanded_h

        # Convert to x1, y1, x2, y2 format
        pred_boxes = torch.zeros_like(deltas)
        pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
        pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
        pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
        pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

        return pred_boxes

    def filter_anchors(self, anchors, objectness_score, nms_thresh=0.7, pre_nms_top_n=2000, post_nms_top_n=300):
        print(f"anchors shape: {anchors.shape}")
        print(f"objectness_score shape: {objectness_score.shape}")
        objectness_score = objectness_score.view(-1)
        anchors = anchors.view(-1, 4)

        pre_nms_top_n = min(pre_nms_top_n, objectness_score.nelement())
        print(pre_nms_top_n)

        sorted_idx = torch.argsort(objectness_score, descending=True)
        print(f"sorted_idx shape: {sorted_idx}")
        top_n_idx = sorted_idx[:pre_nms_top_n]
        print(f"Max index in top_n_idx: {top_n_idx.max()}, anchors shape: {anchors.shape}")

        if top_n_idx.max() >= anchors.shape[0]:
            print("Index out of bounds. Cannot proceed.")

        top_n_anchors = anchors[top_n_idx]
        top_n_scores = objectness_score[top_n_idx]
        print(f"top_n_anchors shape: {top_n_anchors.shape}")
        print(f"top_n_scores shape: {top_n_scores.shape}")

        keep = nms(top_n_anchors, top_n_scores, nms_thresh)
        keep = keep[:post_nms_top_n]

        return keep


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

    def forward(self, x):
        # Backbone feature extraction
        x = self.backbone(x)

        # RPN to generate proposals
        final_boxes, final_scores = self.rpn(x)

        # Assuming single image per batch, creating box_indices tensor
        box_indices = torch.zeros(final_boxes.shape[0], dtype=torch.int64)

        # RoI Align
        rois = self.roi_align(x, final_boxes, box_indices)

        # Classification
        classification = self.classifier(rois)

        # Box Regression
        boxes = self.box_regressor(rois)

        # Mask prediction
        masks = self.mask_branch(rois)

        return classification, boxes, masks




