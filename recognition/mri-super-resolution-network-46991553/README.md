### Abstract
This project implements a brain MRI super-resolution network by training on the ADNI brain dataset. The network is trained to up-scale from 4x down-sampled input images and produce reasonably clear output images.

As proposed by [Shi, 2016](https://arxiv.org/abs/1609.05158), the model utilises efficient sub-pixel convolution by extracting feature maps in the low-resolution space. This reduces computational complexity by waiting until the end of the model to reconstruct to higher dimensions.

### Model Architecture



### Data Processing and Training Procedure
The original MRI images are 240px $\times$ 256px:

<img src="doc/original.png" width="600">

As they are loaded, a Random Horizontal Flip is applied:

<img src="doc/original-flipped.png" width="600">

Once loaded, they are downsampled by a factor of 4 using the Resize() transform:

<img src="doc/downsampled.png" width="600">

These downsampled images are fed into the model, and the model loss is calculated using the original images. The test set is used to generate regular checkpoint outputs during training, and these are later inspected to verify the model isn't getting overfitted.

The model was trained for 10 epochs, which likely that 10 epochs is more than necessary. This can be seen in the loss plot:

<img src="doc/lossplot.png" width="600">

We see a steep drop followed by stability for the rest of the training. Examining the checkpoint images supports this.

The end of epoch 4:

<img src="doc/%5B4,10%5D%5B160,169%5Doutput.png" width="600">

Looks almost identical to the end of epoch 10:

<img src="doc/%5B10,10%5D%5B169,169%5Doutput.png" width="600">

For future model applications, the number of epochs could likely be reduced to decrease training time without significantly impacting model performance.

#### Loss Function - MSE
The MSE (Mean Squared Error) loss function was used with mean reduction:
    $$ℓ(x,y)=\frac{1}{n}\sum^N_{i=1}{l_i}, \text{where }  l_i =(x_i−y_i)^2$$
For tensors $x$ and $y$ with $N$ total elements. MSE is incredibly common for non-classification models, and it is useful for punishing significant outliers in the outputs. In this context those would be significantly different pixels.


#### Optimiser - ADAM
ADAM is well known for its reliable performance in a variety of contexts. It was the first optimiser I tried, with a learning rate of 0.001. Since that produced good results, I didn't test any others - instead I focused my model improvement on architecture changes, activation functions, etc.

### Directory Structure

    data/
    └── AD_NC/
        ├── train/
        │   ├── AD-parent/
        │   │   └── AD/
        │   │       ├── 218391_78.jpeg
        │   │       └── ...
        │   └── NC-parent/
        │       └── NC/
        │           ├── 808819_88.jpeg
        │           └── ...
        └── test/
            ├── AD-parent/
            │   └── AD/
            │       ├── 388206_78.jpeg
            │       └── ...
            └── NC-parent/
                └── NC/
                    ├── 218391_78.jpeg
                    └── ...
    imgs/
    config.py
    dataset.py
    modules.py
    predict.py
    train.py

### Dependencies

This project requires the following packages:

 - pytorch: v2.0.1
 - matplotlib: v3.7.1