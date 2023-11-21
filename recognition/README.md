# COMP3710 Report: Siamese ADNI Model
### Name: Jason Alexander Lukas, Student ID: 46631448
My project aims to fulfill task no. 7, that is to classify Alzheimer's disease in the ADNI dataset using a classifier that is based on the Siamese network. To do this, after defining how to load the ADNI image data in dataset.py, the Siamese model itself in modules.py, and the training routine in train.py, a classifier is defined in predict.py based on the probability of whether two images are of the same class as predicted by the model. If the probability is above 0.5, they are said to be the same, otherwise they are said to be of different classes. Thus, in order to actually use this classifier, an image whose label is known (e.g., from the training set) must be used as an "anchor".

In general, the workflow can be visualised as follows:
![Alt text](image.png)

## Dependencies
The versions required are as follows:
- Keras: 2.10.0
- Tensorflow: 2.10.0
- Python: 3.10.12
- Numpy: 1.25.0

## dataset.py
This file loads the ADNI dataset, assumed to be in the same folder as these files, using Tensorflow (with seeds having been set for reproducibility of results). The test set will be used as the validation set as the test set images appear to already have been split at the patient-level from the training set. This file preprocesses them by following these steps:
1. Convert every image to grayscale to reduce the colour channels and save space.
2. Resize every image to 75x80 px, making sure to preserve the original images' aspect ratios (their original sizes are 240x256 px).
3. For the training set specifically, perform random horizontal flips for data augmentation to prevent overfitting.
4. Normalise the pixel values of every image to be between 0 and 1 for training.

Finally, the function get_data() returns both the loaded training set and test set images for use elsewhere.

## modules.py
Here, the Siamese model's architecture is defined. The architecture is as follows:
1. Inputs: Two images, each of size 75x80 px with 1 colour channel (to match the preprocessed images from dataset.py).
2. CNN Block: The basic CNN block is defined in the get_block() function, containing a Conv2D layer (with 3x3 padding, stride of 1, and L2 Regularisation to prevent overfitting).
3. Depth: A depth of 128 is chosen as the dataset is quite large.
4. CNN, defined as:
    - A CNN Block
    - A 50% Dropout layer to prevent overfitting, forcing the model to learn more robust representations of the data
    - CNN Blocks with twice and quadruple the depth.
    - Another 50% Dropout layer
    - A CNN Block with 8 times the depth
    - Global Average Pooling to reduce the spatial dimension to a single value for every feature map
    - A Dense layer with 64 units and a Sigmoid activation function to capture high-level features and the relationship of the two input images.
5. Concatenation: Onto the feature vectors of the two input images after applying the CNN above to them, concatenation is performed to compare the extracted features from them.
6. Dropout: A final 50% Dropout layer is applied
7. Output Dense: The output Dense layer with 1 unit and a Sigmoid activation to ensure the output is a probability.

## train.py
Here, the Siamese model underlying the classifier is trained on the ADNI dataset. The code is as follows:
1. Load the training and test sets into Numpy arrays.
2. Define the function create_pairs(). This function, an input batch and its labels, will return a Cartesian Product of the input batch with itself with its corresponding label being whether the pair is of the same class.
3. Specify the Checkpoint files and load a Checkpoint (from a previous iteration of training when the model had no L2 Regularisation and Dropout layers and was trained on 4 batches of 40x40 paired images (1600 images)).
4. Compile the model with an Adam optimiser and a Binary Cross Entropy loss (as the model has two classes).
5. For a given training and test set batch, create pairs of the input images and fit the model onto them, saving it as a checkpoint once done.

As mentioned above, the training had already been done once with a previous version of the model for 4 batches. The results of this training are as follows:
![Alt text](image-6.png)
![Alt text](image-7.png)
As can be seen in the 4 graphs above, no improvement had been made. These results can also be found in greater detail in s4663144-siamese-a100_out.txt in my GitHub pull request.

Even after tweaking the model by adding L2 Regularisation and some Dropout layers, no improvement had been forthcoming, with training after one batch resulting in loss: 11.5370 - accuracy: 0.5013 - val_loss: 3.2825 - val_accuracy: 0.5000. Further training was then stopped due to lack of time on my part.

## predict.py
This file contains the code needed to create the classifier for the data based on the constructed Siamese model. The file goes as follows:
1. The first 42 lines are the same as train.py
2. The model is then compiled and a checkpoint after the training on the single batch with the tweaks added to the model is loaded onto the model.
3. A function check_same_class() is defined that accepts two inputs and returns whether or not the model believes they are in the same class.
4. A loop is then defined that iterates through the test set to compare several pairs of images against each other. The (empirical) accuracy is then calculated to gauge the model's performance based on the number of correct and incorrect predictions.

Unfortunately, however, due to the model not improving as seen during training, after testing the model through 1000 images in the test set, the final result (with the seed in modules.py ensuring reproducibility) is:
- Correct Guesses: 496 
- Incorrect Guesses: 504
- Accuracy: 0.496

Thus, the model ultimately has an identical performance to a random guess. The reason for this may be due to the limited number of images the model is trained on or possibly due to other faults in the model's design or other parts of the code itself. Thus, a classifier based on this model would be no better than random guessing, meaning the model unfortunately performs very poorly.