"""
This file can read a model checkpoint and then test it against test set
"""
from dataset import ADNIDataModule
import pytorch_lightning as pl
from train import ViT

# Set up dir of checkpoint
root_dir = './checkpoints/'
# Enter file name below:
file_name = 'ViT.ckpt'

def main():
    checkpoint = root_dir + file_name
    print(checkpoint)

    # Initialise trainer
    trainer = pl.Trainer(max_epochs=1)
    
    # Initialise dataset
    ADNI = ADNIDataModule(batch_size=32, num_workers=0, image_size=[256,240])

    # Set up model from checkpoint
    model = ViT.load_from_checkpoint(checkpoint_path=checkpoint)
    
    # Test model
    trainer.test(model, ADNI)
    print('Test done')

if __name__ == '__main__': main()