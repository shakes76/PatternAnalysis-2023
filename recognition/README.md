# Recognition Tasks
Various recognition tasks solved in deep learning frameworks.

Tasks may include:
* Image Segmentation
* Object detection
* Graph node classification
* Image super resolution
* Disease classification
* Generative modelling with StyleGAN and Stable Diffusion

# Classify Alzheimer's disease of ADNI brain data with perceiver transformer
For this project, I am using the Alzheimer's Disease Neuroimaging Initiative (ADNI) dataset to classify MRI scans of brains that have alzheimers. This project aims to create a perceiver transformer that can detect and classify MRI scans and with over 80% accuracy determine if it has alzheimers. The ADNI dataset has images or the scans with a resolution of 256x240. The model completes a binary classification of the images as a brain can either have or not have alzheimers.

The Perceiver Transformer was proposed in the paper Attention Is All You Need by a team working at the Google https://arxiv.org/pdf/1706.03762.pdf. The Perceiver Transformer was developed with the aim of addressing the limitations of the transformer which was designed for natural languagge processing tasks. The Perceiver Transformer is developed upon the base transformer and it can be given a wider range of datasets to work on from different modalities. The other limitation of the Transform that the Perceiver Transformer build upon is that the quadratic bottleneck which is done through the use of the query, key and values.

Additionally, the Perceiver Transformer utilises a variant of the transformers self attention which results in better training and generalisation.

This is the diagram on the overall architecture of the Perceiver Transformer.

# Image here

# Implmentation
The loss function used was the Cross Entropy Loss function.

For the Optimiser the Adam optimiser was used. This is because of its general purpose functionality and efficiency. This also reduces the risk of the model being overfitted to the training data set. 

The model was run for 40 epochs for training.
The cross attention module ran with a single head according to the original paper.
The self attention module ran four heads and performed self attention four times per head. This was done based on values from the original paper.
The latent dimensions value was 128.
The embedded dimensions value was 32.
The batch size was 5. This was due to hardware limitations I was training the model on.
The model depth was 4 meaning that it performed cross attention and the latent transform 4 times
The learning rate used was 0.0004. Originally I had set 0.004 but the model was not accurate during testing with this parameter so the learning rate was reduced and the accuracy improved.

LATENT_DIMENTIONS = 128
EMBEDDED_DIMENTIONS = 32
CROSS_ATTENTION_HEADS = 1
SELF_ATTENTION_HEADS = 4
SELF_ATTENTION_DEPTH = 4
MODEL_DEPTH = 4
EPOCHS = 2
BATCH_SIZE = 5
LEARNING_RATE = 0.0004 

# Results
After training the model for 40 Epochs it achieved 51% accuracy on the testing set but appears to be random guessing of whether the image is of Alzheimers or not. To resolve this problem I attempt to resolve it by changing the transforms I was using as initially I was randomcropping which could be giving partial segments of the brain however this only slightly improved the accuracy. I also altered the hyperparameters used but this had little effect on the outcome. 
# Images here

The loss during training fluctuates around 50% and does not change much during the training. 

The accuracy of the model is poor being 50% and after further investigation the model appears to be always guessing that the image is an Alzheimers image. This explains why the accuracy is poor as it is always guessing the same result regardless of the image. 