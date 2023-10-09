#Imports
import torch
import torch.optim as optim
import torch.nn as nn


#Hyperparameter
#Currently example paramaters need to find good values
LATENT_DIMENTIONS = 128 # Dimensions in higher dimensionality space
EMBEDDED_DIMENTIONS = 32 # Dimensions in lower dimensionality space
SELF_ATTENTION_DEPTH = 4
CROSS_ATTENTION_HEADS = 1
SELF_ATTENTION_HEADS = 4
EPOCHS = 2
BATCH_SIZE = 10
LEARNING_RATE = 0.004

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
perceiver = 1