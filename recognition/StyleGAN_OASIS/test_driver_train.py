# -*- coding: utf-8 -*
"""
File: test_driver_train.py

Purpose: A driver script that trains a StyleGAN on the OASIS Brain dataset

@author: Peter Beardsley
"""

from train import train_stylegan_oasis

# Run StyleGAN that is preconfigured for OASIS. Set is_rangpur=True if on Rangpur
train_stylegan_oasis(is_rangpur=False)