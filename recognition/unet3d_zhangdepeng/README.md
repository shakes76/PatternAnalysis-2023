
# Segment theProstate 3D data set with the UNet3D

## Task description:
Segment the (downsampled) Prostate 3D data set with the UNet3D. 
**problem** : how to auto segment acquired 3D MR images to six different classes:: background (1), bladder (2), body (3), bone (4), rectum (5) and prostate (6).

**algorithm** : We the network with annotated 3D MR slices from a
representative training set and can segment non-annotated 3D MR image to six classes .
[! img1.png]

## Requirements
+ monai                          1.2.0
+ torch                          1.9.0
+ numpy                          1.21.2
+ matplotlib                     3.4.3

'''
pip install -r requirements.txt
'''
## Usage
### train
、、、
python train.py --loss /loss_type --dataset_root /path_of_dataset 
、、、
### test
'''
python predict.py --pth /path_of_trained_model --dataset_root /path_of_dataset
'''

## The training process of loss and metric 


## Dataset

### Dataset split:
The total number of dataset is 211. The train/val/test is 174/16/21, thus the ratio is almost 10:1:1. We split the dataset according to the patient ID, so that the sample of val/test dataset is brand-new for trained model.

### Dataset preprocess:
+ normalize: With MR image intensity values being non-standardized, the data acquired needs to be normalized to match the range of values.
+ augmentation: for train set, we use random flip to augment the dataset.
+ resize: for training with batches, we use resized to keep the same size 

## Citation
'''
@inproceedings{cciccek20163d,
  title={3D U-Net: learning dense volumetric segmentation from sparse annotation},
  author={{\c{C}}i{\c{c}}ek, {\"O}zg{\"u}n and Abdulkadir, Ahmed and Lienkamp, Soeren S and Brox, Thomas and Ronneberger, Olaf},
  booktitle={Medical Image Computing and Computer-Assisted Intervention--MICCAI 2016: 19th International Conference, Athens, Greece, October 17-21, 2016, Proceedings, Part II 19},
  pages={424--432},
  year={2016},
  organization={Springer}
}
'''

