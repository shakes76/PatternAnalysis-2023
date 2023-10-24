from dataset import isicData
from torch.utils.data import DataLoader, random_split
from torch.optim import SGD
from module import loadModel
import torch
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_data = isicData("./dataset/ISIC-2017_Training_Data", "./dataset/ISIC-2017_Training_Part1_GroundTruth", "./dataset/ISIC-2017_Training_Part3_GroundTruth.csv",)
test_data = isicData("./dataset/ISIC-2017_Test_v2_Data", "./dataset/ISIC-2017_Test_v2_Part1_GroundTruth","./dataset/ISIC-2017_Test_v2_Part3_GroundTruth.csv",)

train_data = torch.utils.data.Subset(train_data, range(50))
train_loader = DataLoader(train_data, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
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

model.eval()  # Set the model to evaluation mode
iou_list = []

with torch.no_grad():
    for images, targets in test_loader:
        images = [img.to(device) for img in images]
        
        # Get model predictions
        predictions = model(images)
        
        # Iterate over predictions and targets to compute IoU for each image
        for pred, target in zip(predictions, targets):
            # Assuming binary masks, threshold predictions at 0.5
            pred_masks = (pred['masks'] > 0.5).squeeze(1).cpu().numpy().astype(int)
            target_masks = target['masks'].cpu().numpy().astype(int)
            
            # You might have multiple masks per image, compute IoU for each and average
            # This approach takes the max IoU for each target, modify as necessary for your needs
            ious = [compute_iou(pred_masks[i], target_masks[j]) for i in range(pred_masks.shape[0]) for j in range(target_masks.shape[0])]
            
            if ious:
                max_iou = max(ious)
                iou_list.append(max_iou)

# Compute mean IoU over validation set
mean_iou = sum(iou_list) / len(iou_list)
print(f"Mean IoU: {mean_iou}")