# Classification of Alzheimer's Disease with Vision Transformer

## ADNI Dataset

The ADNI Dataset consists of brain MRIs of patients with Alzheimers, as well as MRIs of healthy patients. A binary classification exercise is proposed to detect Alzheimers in patients from brain MRI image alone.

![Sample Alzheimers Image](images/ad_sample.jpeg) ![Sample Normal Image](images/nc_sample.jpeg)

**Sample Brain MRI Image of Alzheimers Patient (Left) and Healthy Patient (Right)**

## Vision Transformers

Vision transformers are able to effectively process images by splitting them into many small, "bite-sized" patches. While these patches are processed individually, the positional context of each patch within the original image is preserved with positional embeddings. Vision transformers can produce a variety of useful outputs, including segmentation, classification and image-generation. In the case of this report, a binary classification result is required. This is achieved by the addition of a small multi-layer perceptron (neural) network to the Vision Transformer output. Vision transformers trained on large datasets have shown advanced performance on CNNs by a small margin, which have historically been the preferred choice for computer-vision tasks. 

![Vision Transformer Architecture](images/generic_tfms.png)

## Model Training

30000 images are provided for model training. A split of 70% training images and 30% testing images is chosen. This split provides an optimal trade-off between the maximising the accuracy of the test evaluation, while allowing as much breadth of training data as possible.

As pre-processing, the negative space is cropped from the borders of each image, which has the effect of focusing the model on the brain only as well as rescaling and centring the brains in-frame. Each image is analysed in patches.

![Cropped Brain](images/crop.png) ![Patched Brain](images/patch.png)

**Cropping Negative Space and Splitting Image into Patches**

The model was trained on the training set for 40 epochs, and test error was evaluated at each step. The plot of loss and accuracy for testing and training sets were recorded throughout training.
