# Siamese Network classification on the ADNI brain data set
___
Task: The given task was to "create a classifier based on the Siamese network to classify either Alzheimer’s disease (normal and AD)
of the ADNI brain data set or classify the ISIC 2020 Kaggle Challenge data set (normal and melanoma)
having an accuracy of around 0.8 on the test set." 

As such, the ADNI dataset was chosen for classification. 

![AD and NC Images](./Images/ADNC.png)
<p align="center">
    <em> Figure 1: AD and NC ADNI Images  </em>
</p>

## ADNI Dataset 
___
The ADNI dataset used contains two folders for training and testing respectively, both of which contain a further two subfolders 
for 'NC' (Normal Cognitive) and 'AD' (Alzheimer's Disease). The training set consists of 11120 'NC' images and  10400 'AD' images while 
the testing set consists of 4460 'AD' images and 4540 'NC' images. 

In the dataset, each image is one of 20 slices of a patient's brain and are named according to the patient and the slice of the brain.
i.e. '505732_80' for the 80th slice of patient 505732's brain. 

Shown below in Figure 1 are example images of an 'AD' and a 'NC' brain slice. 

## Siamese Network
___

For this particular implementation of the Siamese Network, a triplet loss Siamese Network was used. Triplet loss works by designating one image an 'anchor' and then selecting a similar image to be the 'positive'
and a dissimilar image to be the 'negative'. Using these three images, triplet loss tries to maximise the distance between the anchor
and the negative image while minimising the distance between the anchor and the positive image. 

It does this with the equation:

> L(a, p, n) = max{d(a_i, p_i) - d(a_i, n_i) + margin, 0}

Where a is the embeddings of the anchor, p is the embeddings of the positive and n is the
embeddings of the negative  (Mandal, S)[^1].  

As seen below in Figure 2, after training the negative moves further from the anchor while the positive moves closer.

![Showcasing how Triplet Loss changes with training](./Images/TripletLossTraining.png)
<p align="center">
    <em> Figure 2: Triplet Loss Training  </em>
</p>

Regarding the specific task, if the anchor image is 'AD', the positive would be 'AD' and the negative 'NC', and vice versa for an 'NC' anchor. 

## Model and Training

The 'backbone' model implemented is a custom 3D resnet that takes 64 input channels and eventually outputs 1000 features. The classifier then takes these 1000 features, passes
them through some fully connected layers and produces 1 output channel which can then be used to classify the image as either 1 representing 'AD' or 0 representing 'NC'. 



Using the triplet loss function mentioned before and the 3D resnet model, the data visualisation shown in Figure 3 was produced after training the resnet for 30 epochs. 

![Visualisation of Features](./Images/data_plot_30Epocha100.png)
<p align="center">
    <em> Figure 3: Feature Visualisation -- 30 Epochs  </em>
</p>

## References:

[^1]: Mandal, S. (2023) *Power of Siamese Networks and Triplet Loss: Tackling Unbalanced Datasets*. Medium.com.
Retrieved from: https://medium.com/@mandalsouvik/power-of-siamese-networks-and-triplet-loss-tackling-unbalanced-datasets-ebb2bb6efdb1

