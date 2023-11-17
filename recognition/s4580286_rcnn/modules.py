
'''
Loads Mask-RCNN model, pretrained on COCO Dataset

'''

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torch.utils.data
import torchvision.models.segmentation
import torch

def load_model():

  #Model
  model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
  in_features = model.roi_heads.box_predictor.cls_score.in_features  # number of input features
  model.roi_heads.box_predictor = FastRCNNPredictor(in_features,num_classes=3)  # replace the pre-trained head with a new one

  #Mask
  in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels  # number of input features
  hidden_layer = 256
  model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       3)
  return model


