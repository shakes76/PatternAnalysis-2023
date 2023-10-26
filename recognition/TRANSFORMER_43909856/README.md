# ViT Transformer for image classification of the ADNI dataset

 
## Description
Include a description of the algorithm and the problem that it solves (a paragraph or so).

Include how the model works (in a paragraph or so).

TODO Include a figure/visualisation of the model.


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
- **TODO** add list of GPU node specs here


## Examples
Provide example inputs and outputs. 

TODO Provide plots of the algorithm (train and validation loss, validation accuracy)
TODO state the achieved test set accuracy and optimal # of epochs


## Preprocessing
Describe any specific preprocessing used (if any) with references. 

### Train, validation, and test splits
The data was split into a training, validation, and test set. 
Training set data was used to train the model, with binary cross-entropy loss used
to evaluate its performance throughout the training process.

During training, the model performance was evaluated on the validation set at the final step of
each epoch. The relationship between training set loss and validation set loss was observed, to note the 
points of training in which the model was overfitting or underfitting. The most optimal 
length of time for training the model was manually selected, and the model was re-trained with a
different number of epochs. As such, validation set performance was used to perform tuning/selection of
a hyperparameter (the number of epochs).

The test set was used to evaluate the model performance on unseen data, quantified
by the accuracy metric.

#### Number of data points
The ADNI dataset contains 1526 patients (30520 MRI image slices).
The test set was composed of data points sampled from the 'test' directory of the ADNI dataset.
This set contained MRI image slices from 223 AD patients (4460 images) and
227 NC patients (4540 images), giving 450 patients (9000 images total). The test set
comprises of roughly 29-30% of the entire dataset.

Training and validation sets contained points sampled from the 'train' directory of this dataset,
which contains around 70-71% of the data.
80% of 'train' dir data (860 patients, 17200 images) was used in the training set, 
with 416 AD patients (8320 images) and 444 NC patients (8880 images).
20% (216 patients, 4320 images) of this data was used in the validation set - 
this contained 104 AD patients (2080 images) and 112 NC patients (2240 images).

#### Justification
The validation set was chosen to be approximately half the size of the test set.
It was considered more beneficial to quantify the model's performance on
a larger selection of unseen data (in the test set), than to utilise more of this
data for the purpose of hyperparameter tuning or training. When more data is moved to the
test set, the distribution of test data more accurately represents the 
characteristics of the entire dataset. 
The size of the training set was also not decreased in favour of the other sets used,
to allow for the model to train on an appropriate quantity of varying data points.

The split between the training and validation sets was stratified 
(attempting to roughly preserve the class proportions within each split set).
A stratified split can result in more effective training/useful testing. In saying
this, I don't believe that this would make a significant difference to this model
(as the class proportions are almost approximately equal).

#### Preventing data leakage
The training and validation set data appears to be independent from the test set data,
with no overlapping patient MRI images within both of the 'train' and 'test' data
directories.

To prevent data leakage within the train and validation sets (split from the 'train'
directory data), the MRI slices were grouped by patient, then the patients were 
shuffled and split between each set. After the split, data points (each MRI image 
slice) were shuffled within each set. This process ensured that the data was 
appropriately shuffled, whilst preventing images from one patient being allocated 
to both the train and validation set.

