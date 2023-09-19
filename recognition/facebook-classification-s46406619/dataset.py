import numpy as np
import os

# set working directory
os.chdir('C:/Area-51/2023-sem2/COMP3710/PatternAnalysis-2023/recognition/facebook-classification-s46406619')

# import raw data files
facebook = np.load('facebook.npz')
edges = facebook['edges']
features = facebook['features'] # X dataset
target = facebook['target'] # y dataset

print('edges:', edges.shape, 'features:', features.shape, 'target:', target.shape)