from dataset import isicData
from torch.utils.data import DataLoader, random_split
from torch.optim import SGD
from module import loadModel
import torch
import torchvision
from torchvision.ops import box_iou


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_data = isicData("./dataset/ISIC-2017_Training_Data", "./dataset/ISIC-2017_Training_Part1_GroundTruth", "./dataset/ISIC-2017_Training_Part3_GroundTruth.csv",)
test_data = isicData("./dataset/ISIC-2017_Test_v2_Data", "./dataset/ISIC-2017_Test_v2_Part1_GroundTruth","./dataset/ISIC-2017_Test_v2_Part3_GroundTruth.csv",)

train_data = torch.utils.data.Subset(train_data, range(50))
train_loader = DataLoader(train_data, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
test_data = torch.utils.data.Subset(test_data, range(50))
test_loader = DataLoader(test_data, batch_size=4, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

model = loadModel()
model.to(device)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

num_epochs = 1

for epoch in range(num_epochs):
    model.train()
    count = 0
    for images, targets in train_loader:
        print(count, len(train_loader))
        count+=1
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

def compute_iou(mask1, mask2):
    """Compute the Intersection over Union (IoU) of two masks."""
    intersection = (mask1 * mask2).sum().item()
    union = (mask1 + mask2).sum().item() - intersection
    return intersection / union

iou_values = []

model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    count = 0
    for images, targets in test_loader:
        print(count, len(test_loader))
        count += 1
        images = [img.to(device) for img in images]
        
        # Assuming the output gives you bounding boxes; modify as needed
        preds = model(images)
        
        # Extract the predicted and ground truth boxes
        # Note: Adjust this depending on the structure of your outputs and targets
        preds_boxes = [pred['boxes'] for pred in preds]
        targets_boxes = [target['boxes'] for target in targets]

        # Compute IoU values for each image in the batch
        for pred_box, target_box in zip(preds_boxes, targets_boxes):
            iou = box_iou(pred_box, target_box)
            # Here, if you have multiple predicted boxes and target boxes, 
            # you might want to determine a way to match them, 
            # for example, by taking the maximum IoU value.
            # For simplicity, we take the mean IoU value.
            iou_values.append(iou.mean().item())

average_iou = sum(iou_values) / len(iou_values)
print(f'Average IoU: {average_iou:.4f}')