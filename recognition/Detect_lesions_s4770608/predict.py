import argparse
import os
from datetime import datetime

import torch
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.transforms import v2, InterpolationMode

import wandb
import torch.nn as nn
import torch.nn.functional as F
from dataset import  CustomISICDataset
from torch.utils.data import DataLoader
import tqdm
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from shapely.geometry import Polygon

import torchvision
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torch.optim as optim
import numpy as np
import wandb
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from main import  select_best_prediction,compute_accuracy,calculate_iou_bbox,log_predictions_to_wandb
from  modules import  get_model_instance_segmentation,ImageClassifier,get_deeplab_model
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--output_path', default="output")
args = parser.parse_args()
os.makedirs(args.output_path,exist_ok=True)
print(f'save figs to dir {args.output_path}')
def calculate_iou(box_1, box_2):
    poly_1 = Polygon([(box_1[0], box_1[1]), (box_1[2], box_1[1]), (box_1[2], box_1[3]), (box_1[0], box_1[3])])
    poly_2 = Polygon([(box_2[0], box_2[1]), (box_2[2], box_2[1]), (box_2[2], box_2[3]), (box_2[0], box_2[3])])
    iou = poly_1.intersection(poly_2).area / poly_1.union(poly_2).area
    return iou


# Load pre-trained Mask R-CNN model

maskrcnn_model = get_model_instance_segmentation(4).cuda()

backbone = maskrcnn_model.backbone.body

# 创建图片分类器
image_classifier = ImageClassifier(backbone, num_classes=3).cuda()  # amodel.eval()
maskrcnn_model.load_state_dict(torch.load('/home/ubuntu/works/code/working_proj/2023S3Course/COMP3710/project/code/PatternAnalysis-2023/recognition/Detect_lesions_s4770608/save_weights/2023-10-19_17-03-26/epoch34.pt'))
maskrcnn_model.eval()

# Load images
test_img_dir = '/home/ubuntu/works/code/working_proj/2023S3Course/COMP3710/project/data/test_imgs'
test_gt_dir = '/home/ubuntu/works/code/working_proj/2023S3Course/COMP3710/project/data/test_gt'
target_size= 384
val_transform_stage1_for_img_mask = transforms.Compose([
    v2.ToTensor(),
])

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]
val_transform_stage2_for_img = transforms.Compose([
    v2.Normalize(mean=imagenet_mean, std=imagenet_std)
])


def collate_fn(batch):
    images, targets = zip(*batch)
    images = [img.cuda() for img in images]

    for target in targets:
        target["boxes"] = target["boxes"].cuda()
        target["labels"] = target["labels"].cuda()
        target["masks"] = target["masks"].cuda()

    return list(images), list(targets)
test_data = CustomISICDataset \
    (csv_file='/home/ubuntu/works/code/working_proj/2023S3Course/COMP3710/project/data/test_label.csv',
     img_dir='/home/ubuntu/works/code/working_proj/2023S3Course/COMP3710/project/data/test_imgs',
     mask_dir='/home/ubuntu/works/code/working_proj/2023S3Course/COMP3710/project/data/test_gt',
     transform_stage1_for_img_mask=val_transform_stage1_for_img_mask,
     transform_stage2_for_img=val_transform_stage2_for_img,
     target_size=target_size)


test_data_loader = DataLoader(test_data, batch_size=1, shuffle=False, collate_fn=collate_fn)
def log_predictions_to_output(images, predictions, targets,predicted_label,output_dir):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    input_image = images[0].cpu().permute(1,2,0).numpy()
    axs[0].imshow(input_image)
    axs[0].set_title("Input Image")
    axs[1].imshow(input_image)
    pred=predictions[0]
    pred_boxes = pred['boxes'].cpu().numpy()
    pred_masks =  pred['masks'].cpu().numpy()
    label_map = {
        0: "Melanoma",
        1: "Seborrheic Keratosis",
        2: "Healthy"
    }
    for box, mask in zip(pred_boxes, pred_masks):
        xmin,ymin, xmax, ymax,  = box
        rect = plt.Rectangle((xmin, ymin), xmax - xmin,ymax - ymin,  fill=False, color='red')
        axs[1].add_patch(rect)
        axs[1].text(xmin, ymin, label_map[predicted_label.item()], color='red')
        axs[1].imshow(mask[0], alpha=0.7)
    axs[1].set_title("Predicted Boxes and Masks")

    # 绘制真实框和掩膜
    axs[2].imshow(input_image)
    targets = targets[0]
    true_boxes = targets['boxes'].cpu().numpy()
    true_masks = targets['masks'].cpu().numpy()
    true_labels = targets['labels'].cpu().numpy()

    for box, mask, label in zip(true_boxes, true_masks, true_labels):
        xmin,ymin, xmax, ymax,  = box
        rect = plt.Rectangle((xmin, ymin),   xmax - xmin,ymax - ymin,fill=False, color='green')
        axs[2].add_patch(rect)
        axs[2].text(xmin, ymin, label_map[label.item()-1], color='green')
        axs[2].imshow(mask, alpha=0.7)
    axs[2].set_title("True Boxes and Masks")


    for ax in axs:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir,f'{i}.png'))


with torch.no_grad():
    pbar_val = tqdm.tqdm(test_data_loader, ' Test', leave=False)
    wandb_images = []
    all_ious=[]
    all_accuracies=[]
    for i, (images, targets) in enumerate(pbar_val):
        predictions = maskrcnn_model(images)
        if len(predictions[0]['boxes']) == 0:
            all_ious.append(0)
            all_accuracies.append(0)
            print('zero prediction occurs')
            continue
        predictions = select_best_prediction(predictions)
        iou = calculate_iou_bbox(predictions[0]["boxes"].cpu().numpy()[0], targets[0]["boxes"].cpu().numpy()[0])
        all_ious.append(iou)
        classify_result = image_classifier(torch.stack(images)).argmax(1)
        labels = torch.tensor([t['labels'] - 1 for t in targets]).cuda()
        accuracy = compute_accuracy(classify_result, labels)
        print(classify_result, labels)
        all_accuracies.append(accuracy)
          # Log images every 10 epochs
        log_predictions_to_output(images, predictions, targets=targets, predicted_label=classify_result,output_dir=args.output_path)
    mean_iou = sum(all_ious) / len(all_ious)
    mean_accuracy = sum(all_accuracies) / len(all_accuracies)
print({"Val Mean IoU": mean_iou, "Val Mean Accuracy": mean_accuracy})
