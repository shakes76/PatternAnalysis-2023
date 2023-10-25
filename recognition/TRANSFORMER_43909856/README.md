# ViT Transformer for image classification of the ADNI dataset

## Description
Include a description of the algorithm and the problem that it solves (a paragraph or so).

Include how the model works (in a paragraph or so).

Include a figure/visualisation of the model.

## Dependencies
This code is written in Python 3.11.5. 

The following libraries/modules are also used:
- pytorch 2.1.0
- pytorch-cuda 11.8
- torchvision 0.16.0
- torchdata 0.7.0
- matplotlib 3.7.2
- scikit-learn 1.3.0
- einops 0.7.0

It is strongly recommended that these packages are installed within a new conda
(Anaconda/Miniconda) environment, and that the code is run within this environment. 
These libraries can then be installed into the conda environment 
using these lines in the terminal:

```
conda install pytorch torchvision torchaudio torchdata pytorch-cuda=11.8 -c pytorch -c nvidia

conda install matplotlib

conda install scikit-learn

pip install einops
``````

Model training was completed on the UQ Rangpur HPC server, using the [insert GPU name here]
node with the following hardware specifications:
- TODO add list of GPU node specs here

## Examples
Provide example inputs and outputs. 

Provide plots of the algorithm.

## Preprocessing
Describe any specific preprocessing used (if any) with references. 

Justify the training, validation, and testing splits of the data.



### Notes


Datasets:
- MRI data (NIFTI image - .nii files) - can use ITK Snap (for prostate data only) or SMILI to visualise

ADNI data:
- Includes image level labels (Alzheimer's or normal brain)
- JPEG images
- Similar to OASIS dataset
- 10k-20k samples

Avoid pre-trained models, but weights of models can be pre-loaded.

