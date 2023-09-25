"""
Hugo Burton
s4698512
20/09/2023

modules.py
Contains the source code of the components of my model
Each component will be designated as a class
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision.transforms as transforms
import numpy as np


class VectorQuantised(nn.Module):
    """
    Takes a tensor and quantises/discretises it
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: int) -> None:
        super(VectorQuantised, self).__init__()

        self._num_embeddings = num_embeddings
        self._embedding_dim = embedding_dim

        self._embedding = nn.Embedding(
            self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1 /
                                             self._num_embeddings, 1 / self._num_embeddings)

        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        """"""

        pass
