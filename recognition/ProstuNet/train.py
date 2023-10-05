from dataset import *
# Required modules
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib
import tensorflow as tf
import time
import os
import sys
# import pydicom
# import nrrd
import scipy.ndimage
import scipy.misc
# import pickle
# import random
# import skimage 

if sys.version_info[0] != 3:
    raise Exception("Python version 3 has to be used!")

print("Currently using")
print("\t numpy ", np.__version__)
print("\t scipy ", scipy.__version__)
print("\t matplotlib ", matplotlib.__version__)
print("\t tensorflow ", tf.__version__)
# print("\t pydicom ", pydicom.__version__)
# print("\t nrrd ", nrrd.__version__)
# print("\t skimage ", skimage.__version__)

np.random.seed(37) # for reproducibility
data = Data()
