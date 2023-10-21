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

from  modules import  get_model_instance_segmentation,ImageClassifier,get_deeplab_model
from PIL import Image
def get_data_loaders(target_size):
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    def collate_fn(batch):
        images,targets = [],[]
        for i in batch:
            if i == None:
                continue
            else:
                images.append(i[0]),targets.append(i[1])
        # images, targets = zip(*batch)
        images = [img.cuda() for img in images]

        for target in targets:
            target["boxes"] = target["boxes"].cuda()
            target["labels"] = target["labels"].cuda()
            target["masks"] = target["masks"].cuda()


        return list(images), list(targets)
    train_transform_stage1_for_img_mask = transforms.Compose([
        v2.RandomVerticalFlip(p=0.3),
        v2.RandomHorizontalFlip(p=0.3),
        v2.RandomRotation(degrees=(0,180)),
        v2.RandomResizedCrop(size=(1129, 1504)),
        # v2.Resize(size=(target_size,target_size),interpolation=InterpolationMode.NEAREST),
        v2.ToTensor(),
    ])


    train_transform_stage2_for_img = transforms.Compose([
        v2.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])



    train_data = CustomISICDataset(
        csv_file='/home/ubuntu/works/code/working_proj/2023S3Course/COMP3710/project/data/train_label.csv',
        img_dir='/home/ubuntu/works/code/working_proj/2023S3Course/COMP3710/project/data/train_imgs',
        mask_dir='/home/ubuntu/works/code/working_proj/2023S3Course/COMP3710/project/data/train_gt',
        transform_stage1_for_img_mask=train_transform_stage1_for_img_mask,
        transform_stage2_for_img=train_transform_stage2_for_img,
        target_size=target_size)
    val_transform_stage1_for_img_mask = transforms.Compose([
        # v2.Resize(size=(target_size, target_size), interpolation=InterpolationMode.NEAREST),
        v2.ToTensor(),

    ])
    val_transform_stage2_for_img = transforms.Compose([
        v2.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])
    val_data = CustomISICDataset(
        csv_file='/home/ubuntu/works/code/working_proj/2023S3Course/COMP3710/project/data/val_label.csv',
        img_dir='/home/ubuntu/works/code/working_proj/2023S3Course/COMP3710/project/data/val_imgs',
        mask_dir='/home/ubuntu/works/code/working_proj/2023S3Course/COMP3710/project/data/val_gt',
        transform_stage1_for_img_mask=val_transform_stage1_for_img_mask,
        transform_stage2_for_img=val_transform_stage2_for_img,
        target_size=target_size)

    test_data = CustomISICDataset\
        (csv_file='/home/ubuntu/works/code/working_proj/2023S3Course/COMP3710/project/data/test_label.csv',
                                  img_dir='/home/ubuntu/works/code/working_proj/2023S3Course/COMP3710/project/data/test_imgs',
                                  mask_dir='/home/ubuntu/works/code/working_proj/2023S3Course/COMP3710/project/data/test_gt',
                                  transform_stage1_for_img_mask=val_transform_stage1_for_img_mask,
                                  transform_stage2_for_img=val_transform_stage2_for_img,
                                  target_size=target_size)
    train_data_loader = DataLoader(train_data, batch_size=4, shuffle=True,collate_fn=collate_fn)
    val_data_loader = DataLoader(val_data, batch_size=1, shuffle=False,collate_fn=collate_fn)
    test_data_loader = DataLoader(test_data, batch_size=1, shuffle=False,collate_fn=collate_fn)
    print(f'Training data: {len(train_data_loader.dataset)} samples, '
          f'{len(train_data_loader)} batches')
    print(f'Validation data: {len(val_data_loader.dataset)} samples, '
          f'{len(val_data_loader)} batches')
    print(f'Testing data: {len(test_data_loader.dataset)} samples, '
          f'{len(test_data_loader)} batches')
    return  train_data_loader,val_data_loader,test_data_loader

def calculate_iou_bbox(box_1, box_2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters:
    box_1, box_2: list of float
        Bounding boxes coordinates: [x1, y1, x2, y2]

    Returns:
    float
        IoU value
    """
    # 将 bounding boxes 从 [x1, y1, x2, y2] 格式转换为 Polygon 所需的坐标格式
    poly_1 = Polygon([(box_1[0], box_1[1]), (box_1[2], box_1[1]), (box_1[2], box_1[3]), (box_1[0], box_1[3])])
    poly_2 = Polygon([(box_2[0], box_2[1]), (box_2[2], box_2[1]), (box_2[2], box_2[3]), (box_2[0], box_2[3])])

    # 计算 IoU
    iou = poly_1.intersection(poly_2).area / poly_1.union(poly_2).area

    return iou







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

def log_predictions_to_wandb(images, predictions, targets,predicted_label):
    plt.close()
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

    return  wandb.Image(plt)



def main():
    # Define your transformations
    max_epoch =50

    target_size = 384
    train_data_loader,val_data_loader,test_data_loader = get_data_loaders(target_size)
    wandb.init(project='ISIC',name='new maskrcnnv2 all requires_grad and classifier no resize ')  # Please set your project and entity name
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    output_folder = os.path.join('save_weights', timestamp)
    os.makedirs(output_folder,exist_ok=True)

    maskrcnn_model = get_model_instance_segmentation(4)

    backbone = maskrcnn_model.backbone.body

    # 创建图片分类器
    image_classifier = ImageClassifier(backbone, num_classes=3)  # a
    maskrcnn_model.cuda()
    image_classifier.cuda()
    image_classifier_loss = torch.nn.CrossEntropyLoss()
    params = [p for p in maskrcnn_model.parameters() if p.requires_grad]
    params.extend([p  for p in image_classifier.parameters() if p.requires_grad])
    optimizer = optim.AdamW(params, lr=0.0005)
    lr_sheduler = CosineAnnealingWarmRestarts(optimizer,T_0=3,T_mult=1,eta_min=2e-7)
    # for cur_e in pbar:
    pbar = tqdm.tqdm(range(max_epoch))
    scaler = torch.cuda.amp.GradScaler()

    max_iou = 0
    for epoch in pbar:
        maskrcnn_model.train()
        image_classifier.train()
        epoch_loss = 0
        if epoch == 0:
            train_data_loader = tqdm.tqdm(train_data_loader)
        else:
            pass
        epoch_loss_dict = {"loss_classifier": 0,
            "loss_box_reg": 0,
           'new_loss_classifier':0,
            "loss_mask": 0,
            "loss_objectness": 0,
            "loss_rpn_box_reg": 0
        }
        for images, targets in train_data_loader:
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                loss_dict = maskrcnn_model(images, targets)
                classify_logits = image_classifier(torch.stack(images))
            labels = torch.tensor([t['labels'] - 1 for t in targets]).cuda()
            classify_loss = image_classifier_loss(classify_logits,labels)
            loss_dict['new_loss_classifier'] = classify_loss

            epoch_loss_dict['new_loss_classifier'] += classify_loss
            losses = sum(loss for loss in loss_dict.values())
            scaler.scale(losses).backward()

            # losses.backward()
            # optimizer.step()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += losses.item()
            for t in loss_dict:
                epoch_loss_dict[t] +=loss_dict[t].item()
            # break

        # wandb.log({"new_loss_classifier": loss_dict['new_loss_classifier'].item()},step=epoch)


        wandb.log({"loss_classifier": epoch_loss_dict['loss_classifier']/2000,
            "loss_box_reg": epoch_loss_dict['loss_box_reg']/2000,
           'new_loss_classifier':epoch_loss_dict['loss_classifier']/2000,
            "loss_mask": epoch_loss_dict['loss_mask']/2000,
            "loss_objectness": epoch_loss_dict['loss_objectness']/2000,
            "loss_rpn_box_reg": epoch_loss_dict['loss_rpn_box_reg']/2000
        }, step=epoch)
        # wandb.log({ "loss_box_reg": loss_dict['loss_box_reg'].item(),
        #            'new_loss_classifier':loss_dict['loss_classifier'].item(),
        #     "loss_mask": loss_dict['loss_mask'].item(),
        #     "loss_objectness": loss_dict['loss_objectness'].item(),
        #     "loss_rpn_box_reg": loss_dict['loss_rpn_box_reg'].item(),
        #     'lr':lr_sheduler.get_lr()
        # }, step=epoch)
        maskrcnn_model.eval()
        image_classifier.eval()

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
                iou = calculate_iou_bbox(predictions[0]["boxes"].cpu().numpy()[0], targets[0]["boxes"].cpu().numpy()[0])
                all_ious.append(iou)
                classify_result = image_classifier(torch.stack(images)).argmax(1)
                labels = torch.tensor([t['labels'] - 1 for t in targets]).cuda()
                accuracy = compute_accuracy(classify_result, labels)
                print( classify_result, labels)
                all_accuracies.append(accuracy)
                if 20<i< 50:  # Log images every 10 epochs
                    wandb_images.append(log_predictions_to_wandb(images,predictions,targets=targets,predicted_label=classify_result))
            wandb.log({"predicted_and_true_boxes_masks": wandb_images},step=epoch)
            mean_iou = sum(all_ious) / len(all_ious)
            mean_accuracy = sum(all_accuracies) / len(all_accuracies)
            torch.save(maskrcnn_model.state_dict(), os.path.join(output_folder, f'epoch{epoch}.pt'))
            if mean_iou > max_iou:
                torch.save(maskrcnn_model.state_dict(),os.path.join(output_folder,'best_iou_model.pt'))
        lr_sheduler.step()
        wandb.log({"Val Mean IoU": mean_iou, "Val Mean Accuracy": mean_accuracy}, step=epoch)
#
# def main_deeplabv3(): TODO Note, this is the traning and validation function for deeplabv3, I put it here because the performance is good
#     # Define your transformations
#     max_epoch =50
#
#     target_size = 384
#     train_data_loader,val_data_loader,test_data_loader = get_data_loaders(target_size)
#     wandb.init(project='ISIC',name='deeplabv3 full ')  # Please set your project and entity name
#     now = datetime.now()
#     timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
#     output_folder = os.path.join('save_weights', timestamp)
#     os.makedirs(output_folder,exist_ok=True)
#
#     deeplabModel = get_deeplab_model(2)
#
#     ce_loss = torch.nn.CrossEntropyLoss()
#
#     # 创建图片分类器
#     params = [p for p in deeplabModel.parameters() if p.requires_grad]
#     optimizer = optim.AdamW(params, lr=0.0005)
#     lr_sheduler = CosineAnnealingWarmRestarts(optimizer,T_0=3,T_mult=1,eta_min=2e-7)
#     pbar = tqdm.tqdm(range(max_epoch))
#     deeplabModel.cuda()
#     max_iou = 0
#     for epoch in pbar:
#         deeplabModel.train()
#         epoch_loss = 0
#         if epoch == 0:
#             train_data_loader = tqdm.tqdm(train_data_loader)
#         else:
#             pass
#         loss_dict=  {'segmentation loss':0,'classification loss':0}
#         for images, targets in train_data_loader:
#             mask_logits,classify_logits = deeplabModel(torch.stack(images).cuda())
#             masks = torch.stack([t['masks']  for t in targets]).cuda()
#             segmentation_loss = ce_loss(mask_logits,masks.squeeze(1).long())
#             loss_dict['segmentation loss'] += segmentation_loss.item()
#             labels = torch.tensor([t['labels'] - 1 for t in targets]).cuda()
#             classification_loss = ce_loss(classify_logits, labels)
#             loss_dict['classification loss']  += classification_loss.item()
#             total_loss = segmentation_loss+ classification_loss
#
#             optimizer.zero_grad()
#             total_loss.backward()
#             optimizer.step()
#
#
#         for t in loss_dict:
#             loss_dict[t] = loss_dict[t]/2000
#         wandb.log(loss_dict, step=epoch)
#
#         deeplabModel.eval()
#
#         all_ious = []
#         all_accuracies = []
#         with torch.no_grad():
#             pbar_val = tqdm.tqdm(val_data_loader, desc=f'Epoch {epoch + 1} VAL', leave=False)
#             wandb_images= []
#             for i,(images, targets) in enumerate(pbar_val):
#                 predictions= dict()
#                 mask_logits, classify_logits = deeplabModel(torch.stack(images).cuda())
#                 predictions['masks'] = mask_logits.argmax(1).unsqueeze(0)
#                 if predictions['masks'].min() == predictions['masks'].max():
#                     continue
#                 predictions['boxes'] =  torchvision.ops.masks_to_boxes(predictions['masks'][0])
#                 iou = calculate_iou_bbox(predictions["boxes"].cpu().numpy()[0], targets[0]["boxes"].cpu().numpy()[0])
#                 all_ious.append(iou)
#                 classify_result = classify_logits.argmax(1)
#                 labels = torch.tensor([t['labels'] - 1 for t in targets]).cuda()
#                 accuracy = compute_accuracy(classify_result, labels)
#                 all_accuracies.append(accuracy)
#                 if 20<i< 50:  # Log images every 10 epochs
#                     predictions=[predictions]
#                     wandb_images.append(log_predictions_to_wandb(images,predictions,targets=targets,predicted_label=classify_result))
#             wandb.log({"predicted_and_true_boxes_masks": wandb_images},step=epoch)
#             mean_iou = sum(all_ious) / len(all_ious)
#             mean_accuracy = sum(all_accuracies) / len(all_accuracies)
#             torch.save(deeplabModel.state_dict(), os.path.join(output_folder, f'epoch{epoch}.pt'))
#             if mean_iou > max_iou:
#                 torch.save(deeplabModel.state_dict(),os.path.join(output_folder,'best_iou_model.pt'))
#         lr_sheduler.step()
#         wandb.log({"Val Mean IoU": mean_iou, "Val Mean Accuracy": mean_accuracy}, step=epoch)

if __name__ == '__main__':
    main()

