import torch
import numpy as np
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"

in_features = 4
out_features = 2
N_layers = 6
