import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from modules import create_classifier
from dataset import load_and_preprocess_data

# Load and preprocess data
Images, Labels = load_and_preprocess_data()

# Split data into training and testing sets
train_indices, test_indices = train_test_split(list(range(Images.shape[0]), train_size=0.8, test_size=0.2, shuffle=True)
x_train = Images[train_indices]
y_train = Labels[train_indices]
x_test = Images[test_indices]
y_test = Labels[test_indices]
