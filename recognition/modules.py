import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


def get_maskrcnn_model(num_classes):
    model = maskrcnn_resnet50_fpn(pretrained=True)

    in_features_box = model.roi_heads.box_predictor.cls_score.in_features
    box_predictor = FastRCNNPredictor(in_features_box, num_classes)
    model.roi_heads.box_predictor = box_predictor

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)
    model.roi_heads.mask_predictor = mask_predictor

    return model
