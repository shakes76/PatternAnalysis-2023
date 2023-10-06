# Pattern Analysis
Pattern Analysis of various datasets by COMP3710 students at the University of Queensland.

We create pattern recognition and image processing library for Tensorflow (TF), PyTorch or JAX.

This library is created and maintained by The University of Queensland [COMP3710](https://my.uq.edu.au/programs-courses/course.html?course_code=comp3710) students.

The library includes the following implemented in Tensorflow:
* fractals 
* recognition problems

In the recognition folder, you will find many recognition problems solved including:
* OASIS brain segmentation
* Classification
etc.

////////////

The following is a solution which segments the ISIC data set using an improved U-Net architecture.
The model strives to achieve a minimum DICE similarity coefficient of 0.8 on the test set for all labels.

The improved U-Net differs from traditional U-Net in the following ways:
- The modified U-Net is designed to process 3D input blocks of dimensions 128x128x128 voxels, enabling it to work with volumetric data common in medical imaging, unlike the traditional U-Net which typically handles 2D images.
- Modules are crafted with specific convolutional layers, dropout layers, and upscaling methods to better handle the requirements of the task
- Modified U-Net employs deep supervision by integrating segmentation layers at different levels of the network, allowing for more refined learning and error propagation through the network
- Specialised DICE loss function used to handle class imbalances
- Extensive data pre-processing (data normalisation then clipping images to a bound and rescaling)
- Survival prediction based of radiomics
- The testing procedure segments an entire patient at once to overcome potential problems with tile-based segmentation

//////////////////

ARCHITECTURE:
- 

PROCESS:

DATA PREPROCESSING:
- Each modality of each patient's data is normalised independently by subtracting the mean and dividing by the standard deviation of the brain region.
- The normalised images are clipped at [−5,5] to remove outliers and are rescaled to [0,1], setting the non-brain region to 0.

Training:
- training data composed of randomly sampled patches size 128 x 128 x 128 voxels with a batch size of 2
- trained over 300 epochs (1 epoch = iteration over 100 batches)
- Adam optimiser used with initial learning rate of 5 * 10^(-4), following a learning rate sec of the initial learning rate * 0.985^(epoch) and a l2 weight decay of 10^(-5)
- Multi-class Dice loss function used
​- A variety of data augmentation techniques are applied on-the-fly during training to prevent overfitting. These include random rotations, random scaling, random elastic deformations, gamma correction augmentation, and mirroring.

Testing:
- At test time, the entire patient data is segmented at once, leveraging the fully convolutional nature of the network
- Test-time data augmentation is performed by mirroring the images and averaging the softmax outputs over several dropout samples
 



