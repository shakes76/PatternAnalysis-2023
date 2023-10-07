import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


def get_maskrcnn_model(num_categories=3):

    maskrcnn_model = maskrcnn_resnet50_fpn(pretrained=True)

    # Adjust the box predictor to match the number of classes
    box_features = maskrcnn_model.roi_heads.box_predictor.cls_score.in_features
    custom_box_predictor = FastRCNNPredictor(box_features, num_categories)
    maskrcnn_model.roi_heads.box_predictor = custom_box_predictor

    # Adjust the mask predictor to match the number of classes
    mask_features = maskrcnn_model.roi_heads.mask_predictor.conv5_mask.in_channels
    custom_mask_predictor = MaskRCNNPredictor(mask_features, 256, num_categories)
    maskrcnn_model.roi_heads.mask_predictor = custom_mask_predictor

    return maskrcnn_model
