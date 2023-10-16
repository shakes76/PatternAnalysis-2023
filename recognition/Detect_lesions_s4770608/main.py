import os
from datetime import datetime

import torch
from torchvision.models.detection.anchor_utils import AnchorGenerator

import wandb
import torch.nn as nn
import torch.nn.functional as F
from dataset import  CustomISICDataset
from torch.utils.data import DataLoader
import tqdm
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

import torchvision
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torch.optim as optim
import numpy as np
import wandb
from PIL import Image
def get_data_loaders(target_size):
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    def collate_fn(batch):
        images, targets = zip(*batch)
        images = [img.cuda() for img in images]

        for target in targets:
            target["boxes"] = target["boxes"].cuda()
            target["labels"] = target["labels"].cuda()
            target["masks"] = target["masks"].cuda()


        return list(images), list(targets)
    train_transform = transforms.Compose([
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomHorizontalFlip(p=0.3),
        # transforms.RandomRotation(degrees=(0,180),p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])


    train_mask_transform = transforms.Compose([
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomHorizontalFlip(p=0.3),
        # transforms.RandomRotation(degrees=(0,180),p=0.3),
        transforms.ToTensor(),
    ])
    train_data = CustomISICDataset(
        csv_file='/home/ubuntu/works/code/working_proj/2023S3Course/COMP3710/project/data/train_label.csv',
        img_dir='/home/ubuntu/works/code/working_proj/2023S3Course/COMP3710/project/data/train_imgs',
        mask_dir='/home/ubuntu/works/code/working_proj/2023S3Course/COMP3710/project/data/train_gt',
        transform=train_transform,
        mask_transoform=train_mask_transform,
        target_size=target_size)
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)

    ])
    val_mask_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    val_data = CustomISICDataset(
        csv_file='/home/ubuntu/works/code/working_proj/2023S3Course/COMP3710/project/data/val_label.csv',
        img_dir='/home/ubuntu/works/code/working_proj/2023S3Course/COMP3710/project/data/val_imgs',
        mask_dir='/home/ubuntu/works/code/working_proj/2023S3Course/COMP3710/project/data/val_gt',
        transform=val_transform,
        mask_transoform=val_mask_transform,
        target_size=target_size)

    test_data = CustomISICDataset(csv_file='/home/ubuntu/works/code/working_proj/2023S3Course/COMP3710/project/data/test_label.csv',
                                  img_dir='/home/ubuntu/works/code/working_proj/2023S3Course/COMP3710/project/data/test_imgs',
                                  mask_dir='/home/ubuntu/works/code/working_proj/2023S3Course/COMP3710/project/data/test_gt',
                                  transform=val_transform,
                                  mask_transoform=val_mask_transform,
                                  target_size=target_size)
    train_data_loader = DataLoader(train_data, batch_size=24, shuffle=True,collate_fn=collate_fn)
    val_data_loader = DataLoader(val_data, batch_size=1, shuffle=False,collate_fn=collate_fn)
    test_data_loader = DataLoader(test_data, batch_size=1, shuffle=False,collate_fn=collate_fn)
    print(f'Training data: {len(train_data_loader.dataset)} samples, '
          f'{len(train_data_loader)} batches')
    print(f'Validation data: {len(val_data_loader.dataset)} samples, '
          f'{len(val_data_loader)} batches')
    print(f'Testing data: {len(test_data_loader.dataset)} samples, '
          f'{len(test_data_loader)} batches')
    return  train_data_loader,val_data_loader,test_data_loader



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


def compute_iou(pred_masks, target_masks):
    intersect = torch.logical_and(pred_masks, target_masks)
    union = torch.logical_or(pred_masks, target_masks)
    iou = torch.sum(intersect.float()) / torch.sum(union.float())
    return iou.item()


def compute_accuracy(pred_labels, target_labels):
    correct = (pred_labels == target_labels).sum().item()
    total = len(target_labels)
    return correct / total


def select_best_prediction(predictions):
    """
    predictions: List of dicts. Each dict contains 'boxes', 'labels', 'scores', and 'masks'.
    """
    best_predictions = []

    for pred in predictions:
        # 获取得分最高的预测框的索引
        if len(pred['scores']) ==0:
            print(len(pred['scores']),len(pred['boxes']))
            max_score_idx=0
        else:
            max_score_idx = pred['scores'].argmax()

        # 选择得分最高的预测框
        best_pred = {
            'boxes': pred['boxes'][max_score_idx].unsqueeze(0),  # 添加额外的维度，以保持一致性
            'labels': pred['labels'][max_score_idx].unsqueeze(0),
            'scores': pred['scores'][max_score_idx].unsqueeze(0),
            'masks': pred['masks'][max_score_idx].unsqueeze(0)
        }
        best_predictions.append(best_pred)

    return best_predictions

def log_predictions_to_wandb(images, predictions, targets,label):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    input_image = images[0]
    axs[0].imshow(images[0].cpu().numpy())
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
        ymin, xmin, ymax, xmax = box
        rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color='red')
        axs[1].add_patch(rect)
        axs[1].text(xmin, ymin, label_map[label], color='red')
        axs[1].imshow(mask, alpha=0.5)
    axs[1].set_title("Predicted Boxes and Masks")

    # 绘制真实框和掩膜
    axs[2].imshow(input_image)
    targets = targets[0]
    true_boxes = targets['boxes'].cpu().numpy()
    true_masks = targets['masks'].cpu().numpy()
    true_labels = targets['labels'].cpu().numpy()

    for box, mask, label in zip(true_boxes, true_masks, true_labels):
        ymin, xmin, ymax, xmax = box
        rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color='green')
        axs[2].add_patch(rect)
        axs[2].text(xmin, ymin, label_map[label], color='green')
        axs[2].imshow(mask, alpha=0.5)
    axs[2].set_title("True Boxes and Masks")


    for ax in axs:
        ax.axis('off')

    plt.tight_layout()
    return  wandb.Image(plt)


#
#
# # WandB Initialization
#
# def log_predictions_to_wandb(images, predictions, targets,step):
#     """
#     Log bounding box predictions to WandB.
#
#     Args:
#         images: List of input images
#         predictions: Model's predictions
#         num_samples: Number of samples to log
#     """
#
#     # Select a subset of images and predictions to log
#     images_to_log = images
#     predictions_to_log = predictions
#     targets_to_log = targets
#
#     def map_label_to_name(label):
#         mapping = {
#             1: "Melanoma",
#             2: "Seborrheic Keratosis",
#             3: "Healthy"
#         }
#         return mapping[label]
#     # Convert model predictions to wandb format
#     wandb_images = []
#     for image, pred, target in zip(images_to_log, predictions_to_log, targets_to_log):
#         # Prepare data for visualization
#         box_data = []
#         predictions_box_data = []
#         ground_truth_box_data = []
#         # Process predictions
#         for box, score, label, mask in zip(
#                 pred['boxes'].cpu().numpy(),
#                 pred['scores'].cpu().numpy(),
#                 pred['labels'].cpu().numpy(),
#                 pred['masks'].cpu().numpy()
#         ):
#             label_name = map_label_to_name(label)
#             predictions_box_data.append({
#                 'position': {
#                     'minX': float(box[0]),
#                     'maxX': float(box[2]),
#                     'minY': float(box[1]),
#                     'maxY': float(box[3]),
#                 },
#                 'class_id': int(label),
#                 'box_caption': f"PRED: {label_name} ({score:.2f})",
#                 'scores': {
#                     'objectivity': float(score),  # Explicitly convert score to Python float
#                     'class_prob': float(score),  # Explicitly convert score to Python float
#                     'class_score': float(score),
#                 },
#                 'domain': 'pixel',
#             })
#
#         # Process ground truth
#         for box, label in zip(target['boxes'].cpu().numpy(), target['labels'].cpu().numpy()):
#             label_name = map_label_to_name(label)
#             ground_truth_box_data.append({
#                 'position': {
#                     'minX': float(box[0]),
#                     'maxX': float(box[2]),
#                     'minY': float(box[1]),
#                     'maxY': float(box[3]),
#                 },
#                 'class_id': int(label),
#                 'box_caption': f"TRUE: {label_name}",
#                 'domain': 'pixel',
#             })
#
#         # Convert image from torch.Tensor to PIL.Image
#         image_pil = transforms.ToPILImage()(image)
#         image_np = np.array(image_pil)
#         if image_np.shape[-1] == 3:
#             alpha_channel = np.ones(image_np.shape[:2] + (1,), dtype=image_np.dtype) * 255
#             image_np = np.concatenate([image_np, alpha_channel], axis=-1)
#         # Overlay mask onto image for predictions
#         for mask in pred['masks'].cpu().numpy():
#             mask = mask[0].astype(np.uint8) * 255  # Assuming single channel mask
#             mask_pil = Image.fromarray(mask).convert("RGBA")
#             mask_np = np.array(mask_pil)
#
#             # Creating a colored mask with 50% transparency
#             mask_colored = np.zeros_like(mask_np)
#             mask_colored[..., :3] = [255, 0, 0]  # Red color mask
#             mask_colored[..., 3] = mask_np[..., 3] * 0.5  # 50% Transparency
#
#             # Overlaying the mask
#             image_np = Image.alpha_composite(Image.fromarray(image_np), Image.fromarray(mask_colored))
#         mapping = {
#             1: "Melanoma",
#             2: "Seborrheic Keratosis",
#             3: "Healthy"
#         }
#         # Log image, bounding box data, and masks to wandb
#         wandb_images.append(wandb.Image(image_np, boxes={
#             'predictions': {
#                 'box_data': predictions_box_data,
#                 'class_labels': mapping,  # Assume you have a list of class names here
#             },
#             'ground_truth': {
#                 'box_data': ground_truth_box_data,
#                 'class_labels': mapping,  # Assume you have a list of class names here
#             }
#         }))
#     return  wandb_images[0]

def main():
    # Define your transformations
    max_epoch =50

    target_size = 224
    train_data_loader,val_data_loader,test_data_loader = get_data_loaders(target_size)
    wandb.init(project='ISIC')  # Please set your project and entity name
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    output_folder = os.path.join('save_weights', timestamp)
    os.makedirs(output_folder,exist_ok=True)

    maskrcnn_model = get_model_instance_segmentation(4)
    backbone = maskrcnn_model.backbone

    # 创建图片分类器
    image_classifier = ImageClassifier(backbone, num_classes=3)  # a
    maskrcnn_model.cuda()
    image_classifier.cuda()
    image_classifier_loss = torch.nn.CrossEntropyLoss()
    params = [p for p in maskrcnn_model.parameters() if p.requires_grad]
    params.extend([p  for p in image_classifier.parameters() if p.requires_grad])
    optimizer = optim.AdamW(params, lr=0.0005, weight_decay=0.0005)

    # for cur_e in pbar:
    pbar = tqdm.tqdm(range(max_epoch))

    max_iou = 0
    for epoch in pbar:
        maskrcnn_model.train()
        image_classifier.train()
        epoch_loss = 0
        if epoch == 0:
            train_data_loader = tqdm.tqdm(train_data_loader)
        else:
            pass
        for images, targets in train_data_loader:
            loss_dict = maskrcnn_model(images, targets)
            classify_logits = image_classifier(torch.stack(images))
            labels = torch.tensor([t['labels'] - 1 for t in targets]).cuda()
            classify_loss = image_classifier_loss(classify_logits,labels)
            loss_dict['new_loss_classifier'] = classify_loss
            losses = sum(loss for loss in loss_dict.values())
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            epoch_loss += losses.item()
            # break

        # wandb.log({"loss_classifier": loss_dict['loss_classifier'].item(),
        #     "loss_box_reg": loss_dict['loss_box_reg'].item(),
        #            'new_loss_classifier':loss_dict['loss_classifier'].item(),
        #     "loss_mask": loss_dict['loss_mask'].item(),
        #     "loss_objectness": loss_dict['loss_objectness'].item(),
        #     "loss_rpn_box_reg": loss_dict['loss_rpn_box_reg'].item()
        # }, step=epoch)
        wandb.log({ "loss_box_reg": loss_dict['loss_box_reg'].item(),
                   'new_loss_classifier':loss_dict['loss_classifier'].item(),
            "loss_mask": loss_dict['loss_mask'].item(),
            "loss_objectness": loss_dict['loss_objectness'].item(),
            "loss_rpn_box_reg": loss_dict['loss_rpn_box_reg'].item()
        }, step=epoch)
        maskrcnn_model.eval()
        image_classifier.eval()

        val_loss = 0
        all_ious = []
        all_accuracies = []
        with torch.no_grad():
            pbar_val = tqdm.tqdm(val_data_loader, desc=f'Epoch {epoch + 1} VAL', leave=False)
            wandb_images= []
            for i,(images, targets) in enumerate(pbar_val):
                predictions = maskrcnn_model(images)
                if  len(predictions[0]['boxes']) ==0:
                    all_ious.append(0)
                    all_accuracies.append(0)
                    print('zero prediction occurs')
                    continue
                predictions = select_best_prediction(predictions)
                # Compute IoU and append to all_ious list
                iou = compute_iou(predictions[0]["boxes"], targets[0]["boxes"].to('cuda'))
                all_ious.append(iou)
                # print(predictions)
                # Compute classification accuracy and append to all_accuracies list

                classify_result = image_classifier(torch.stack(images)).argmax(1)
                labels = torch.tensor([t['labels'] - 1 for t in targets]).cuda()

                accuracy = compute_accuracy(classify_result, labels)
                all_accuracies.append(accuracy)
                if i< 30:  # Log images every 10 epochs
                    wandb_images.append(log_predictions_to_wandb(images,predictions,targets=targets,label=labels))
            wandb.log({"predicted_and_true_boxes_masks": wandb_images[0:30]},step=epoch)
            mean_iou = sum(all_ious) / len(all_ious)
            mean_accuracy = sum(all_accuracies) / len(all_accuracies)
            torch.save(maskrcnn_model.state_dict(), os.path.join(output_folder, f'epoch{epoch}.pt'))
            if mean_iou > max_iou:
                torch.save(maskrcnn_model.state_dict(),os.path.join(output_folder,'best_iou_model.pt'))
        wandb.log({"Val Mean IoU": mean_iou, "Val Mean Accuracy": mean_accuracy}, step=epoch)

if __name__ == '__main__':
    main()

