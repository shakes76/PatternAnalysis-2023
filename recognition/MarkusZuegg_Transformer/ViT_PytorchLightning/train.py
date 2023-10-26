"""
This file contains main ViT class and will create a new instance
of the model and train, test it
"""
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from dataset import ADNIDataModule
from modules import VisionEncoder

class ViT(pl.LightningModule):
    """Pytorch_lightning all encompasing module.

    Handles creation of VisionEncoder and loss calculation
    along with optermiser and lr rate scheduler
    
    """
    def __init__(self, config, lr):
        super().__init__()
        self.save_hyperparameters()
        # Model (neural net)
        self.model = VisionEncoder(**config)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[35, 100], gamma=0.1)
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
    # Image size based from dataset
    image_size = [256,240]

    # Set up hyperparameters
    lr = 3e-5
    ADNI_config = {
        "embed_dim": 256,
        "hidden_dim": 512,
        "num_heads": 8,
        "num_layers": 4,
        "patch_size": 4,
        "num_channels": 3,
        "image_size": image_size,
        "num_classes": 2,
        "dropout": 0.2,}
    
    # Initialise module class
    ViT = ViT(ADNI_config, lr=lr)

    # Set up varibles for DataModule class
    batch_size = 32 #working 16 on hpc
    num_workers = 0 #issue with multiproccessing pytorch_lightning
                    #must use 0
    max_epochs = 30
    
    # Initialise DataModule class
    ADNI = ADNIDataModule(batch_size=batch_size, 
                        image_size=image_size,  
                        num_workers=num_workers)
    
    # Initialise Trainer class
    trainer = pl.Trainer(max_epochs=max_epochs,
                        accelerator='gpu',
                        devices=1)
    
    # Train model (with validation)
    trainer.fit(ViT, ADNI)

    # Test model against test set
    trainer.test(ViT, ADNI)

def main():
    train_model() #runs train_model essentially main

if __name__ == '__main__': main()