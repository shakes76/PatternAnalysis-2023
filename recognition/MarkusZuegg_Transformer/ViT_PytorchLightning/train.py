"""
#! make a file header
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger
from dataset import CIFAR10DataModule, ADNIDataModule
from modules import VisionEncoder

class ViT(pl.LightningModule):
    """Pytorch_lightning all encompasing module.

    Handles creation of VisionEncoder and loss calculation
    along with optermiser and lr rate schedular
    
    """
    def __init__(self, config, lr):
        super().__init__()
        self.save_hyperparameters()
        self.model = VisionEncoder(**config)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
        return [optimizer], [lr_scheduler]

    def _calculate_loss(self, batch, mode="train"):
        imgs, labels = batch
        preds = self.model(imgs)
        loss = F.cross_entropy(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()

        self.log("%s_loss" % mode, loss)
        self.log("%s_acc" % mode, acc)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="test")

def train_model():
    return

def main():

    image_size = [256,240]
    lr = 3e-4
    ADNI_config = {
        "embed_dim": 256,
        "hidden_dim": 512,
        "num_heads": 8,
        "num_layers": 4,
        "patch_size": 8,
        "num_channels": 3,
        "image_size": image_size,
        "num_classes": 2,
        "dropout": 0.2,}
    model = ViT(ADNI_config, lr=lr)

    batch_size = 1 #working 16 on hpc
    num_workers = 0 #num_workers = 0 if windows
    max_epochs = 30

    #! ================remove validation loader ========================

    
    ADNI = ADNIDataModule(batch_size=batch_size, 
                        image_size=image_size,  
                        num_workers=num_workers)
    
    trainer = pl.Trainer(max_epochs=max_epochs,
                        accelerator='gpu',
                        devices=1)
    
    trainer.fit(model, ADNI)
    trainer.test(model, ADNI)

if __name__ == '__main__': main()