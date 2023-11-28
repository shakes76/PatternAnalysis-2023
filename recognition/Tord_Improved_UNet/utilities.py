import torch
import matplotlib.pyplot as plt
import wandb
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, prediction, target):
        intersection = (prediction * target).sum()
        prediction_sum = prediction.sum()
        target_sum = target.sum()
        dice = (2.0 * intersection + self.smooth) / (prediction_sum + target_sum + self.smooth)
        dice_loss = 1 - dice
        return dice_loss
    
    def accuracy(self, prediction, target):
        prediction = (prediction > 0.5).float()
        correct = (prediction==target).sum()
        pixels = torch.numel(prediction)
        accuracy = correct / pixels + 1e-8
        return accuracy
  

    
class Wandb_logger():
    def __init__(self, model, criterion, config):
        super(Wandb_logger, self).__init__()
        wandb.init(project="ISIC2018", config=config)
        wandb.watch(model, criterion, log="all", log_freq=10)

    def log_test(self, accuracy, loss):
        wandb.log({"accuracy": accuracy, 'loss': loss})
        
    
    def print_images(prediction, target):
        plt.close('all')
        image_tensor = prediction.cpu().detach()
        ground = target.cpu().detach()
        image_array = image_tensor.numpy()
        ground_array = ground.numpy()


        plt.figure(figsize=(15, 7))

        plt.subplot(1, 2, 1)
        plt.imshow(image_array[0, 0], cmap='gray')
        plt.title("Prediction")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(ground_array[0, 0], cmap='gray')
        plt.title("Ground Truth")
        plt.axis('off')
        plt.show()

    def log(self, epoch, batch, loss, accuracy):
        wandb.log({"epoch": epoch, 'batch': batch, "loss": loss, "accuracy": accuracy})