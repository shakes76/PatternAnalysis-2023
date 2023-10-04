# Brain MRI Super-Resolution CNN
## COMP3710 Report - Pattern Recognition - Jarrod Mann - s4696612

## Introduction
The project aimed to create a deep learning model that could sufficiently upsample a low
resolution image. Specifically, the project focused on upsampling brain MRI scans. Creating an
effective model for this task would mean less overall storage space would be required for the
scans while they were not actively being used. Instead, a low resolution image could be stored,
and then be processed through the model each time its use was required. Therefore, the model
aims to reconstruct brain MRI scan images to as high a detail as possible.

## Model Implementation