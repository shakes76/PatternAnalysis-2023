"""
This file will contain the source code for training, validating, testing and
saving the model. The model will be imported from modules.py, and the data loader
will be imported from dataset.py. The losses and metrics will be plotted during
training.
"""

from TRANSFORMER_43909856.dataset import *
from TRANSFORMER_43909856.modules import *

"""

Splitting train set data into a training and validation set (8/2 split)
Test set not used for generative models

As we need to split the train dataset into train and validation sets, 
the patient-level split means when you split the validation set you should
 avoid one patient's MRI slices that appear in both the train and validation set. 
 So you should split the train set by the patient ID which is the 390009 in 
 your example and the last two numbers 78 means the slice ID of this 
 patient's MRI image.
 Did you guys try to do patient-level split? Basically you will need to keep 
 the same patient ID (in the filename) in either training or validation set. 
 Otherwise same patients will appear in both training and validation set and 
 the model will simply remember the label of that patient and cannot learn anything.
"""
