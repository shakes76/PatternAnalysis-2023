import torchvision
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
import numpy as np

# def load_model(model_type=0):
#     if model_type == 0:
#         PATH = SIAMESE_MODEL_SAVE_PATH
#     else:
#         PATH = CLASSIFIER_MODEL_SAVE_PATH

#     if os.path.exists(PATH):
#         load_model = torch.load(PATH)
#         epoch = load_model['epochs']
#         model = load_model['model']
#         criterion = load_model['criterion']
#         optimizer = load_model['optimizer']
#         scheduler = load_model['scheduler']

#         return epoch, model, criterion, optimizer, scheduler
#     else:
#         return None