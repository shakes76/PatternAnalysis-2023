#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
from torch import nn



# classes

class FeedForward(nn.Module):
    def __init__(self, x, y):
        super().__init__()

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, x, heads):
        super().__init__()
 


# In[ ]:




