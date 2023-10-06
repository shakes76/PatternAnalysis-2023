Task description:
Segment the (downsampled) Prostate 3D data set with the UNet3D [5] with all labels having a minimum
Dice similarity coefficient of 0.7 on the test set. See also CAN3D [6] for more details and use the data
augmentation library here for TF or use the appropriate transforms in PyTorch. [Normal Difficulty]

dataset.py 
Dataset preprocess:
With MR image intensity values being non-standardized, the data acquired needs to be normalized to match the range
of values and avoid the network’s initial biases

modules.py
implement the simple version of UNET3D
