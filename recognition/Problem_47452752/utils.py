import torch
from random import choice
import torchvision.transforms.functional as TF


class RandomRotate90:
    """Randomly rotates the image by 90, 180, or 270 degrees."""

    def __init__(self, p=1.0):
        self.p = p

    def __call__(self, sample):
        image, mask = sample["image"], sample["mask"]
        if torch.rand(1) < self.p:
            degrees = [90, 180, 270]
            angle = choice(degrees)
            image = TF.rotate(image, angle)
            mask = TF.rotate(mask, angle)
        return {"image": image, "mask": mask}


class RandomCenterCrop:
    """Randomly crops the center of the image by 80% or 70%."""

    def __init__(self, scales=[0.8, 0.7], p=1.0):
        self.scales = scales
        self.p = p

    def __call__(self, sample):
        image, mask = sample["image"], sample["mask"]
        if torch.rand(1) < self.p:
            scale = choice(self.scales)
            image = TF.center_crop(
                image, (int(image.height * scale), int(image.width * scale))
            )
            mask = TF.center_crop(
                mask, (int(mask.height * scale), int(mask.width * scale))
            )
        return {"image": image, "mask": mask}


class DictTransform:
    def __init__(self, transform, transform_mask=True):
        self.transform = transform
        self.transform_mask = transform_mask
    def __call__(self, sample):
        image, mask = sample["image"], sample["mask"]
        if self.transform_mask:
            return {
                "image": self.transform(image),
                "mask": self.transform(mask),
            }
        return {
            "image": self.transform(image),
            "mask": mask,
        }


def dice_loss(predicted, target, epsilon=1e-7):
    # Flatten the tensors 
    predicted = predicted.view(-1)
    target = target.view(-1)

    intersection = (predicted * target).sum()
    union = predicted.sum() + target.sum()

    dice_loss = 2 * (intersection + epsilon) / (union + epsilon)
    return 1 - dice_loss


def general_dice_loss(predicted, target):
    # One-hot encode the target segmentation map
    target_one_hot = torch.zeros_like(predicted)
    for k in range(target_one_hot.shape[1]):
        target_one_hot[:, k] = target == k

    # Compute the Dice loss for each class, then average
    intersection = (predicted * target_one_hot).sum(dim=(2, 3))
    union = (predicted + target_one_hot).sum(dim=(2, 3))

    dice_scores = 2 * intersection / union
    loss = 1 - dice_scores.mean()

    return loss

def dice_coefficient(predicted, target):
    """ Compute Dice Coefficient for single sample. """
    intersection = (predicted * target).sum().float()
    union = (predicted + target).sum().float()
    
    if union == 0:
        return 1.0

    return 2 * intersection / union