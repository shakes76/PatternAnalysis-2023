import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from modules import create_classifier
from dataset import load_and_preprocess_data

# Load and preprocess data
Images, Labels = load_and_preprocess_data()