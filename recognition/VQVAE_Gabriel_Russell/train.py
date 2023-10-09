"""
Created on Monday Sep 18 12:20:00 2023

This script is for training, validating, testing and saving the VQVAE model.
The model is imported from modules.py and the data loader is imported
from dataset.py. Appropriate metrics are plotted during training.

@author: Gabriel Russell
@ID: s4640776

"""

from modules import *
from dataset import *
import train_VQVAE
import numpy as np
import torch
import matplotlib.pyplot as plt

train_VQVAE_model = train_VQVAE.TrainVQVAE()

train_VQVAE_model.train()

#Load VQVAE Model

#Make encoding 


