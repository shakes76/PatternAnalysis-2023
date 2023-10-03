# Classification of Alzheimer's Disease with Vision Transformer

## ADNI Dataset

The ADNI Dataset consists of brain MRIs of patients with Alzheimers, as well as MRIs of healthy patients. A binary classification exercise is proposed to detect Alzheimers in patients from brain MRI image alone.

![Sample Alzheimers Image](images/ad_sample.jpeg) ![Sample Normal Image](images/nc_sample.jpeg)

**Sample Brain MRI Image of Alzheimers Patient (Left) and Healthy Patient (Right)**

30000 images are provided for model training. A split of 70% training images and 30% testing images is chosen. 

## Vision Transformers

The negative space is cropped from the borders of each image, which has the effect of focusing the model on the brain only as well as rescaling and centring the brains in-frame. Each image is analysed in patches.

![Cropped Brain](images/crop.png) ![Patched Brain](images/patch.png)

**Cropping Negative Space and Splitting Image into Patches**

## Model Training
