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


Process:

Data Preprocessing:
- Normalization is performed to standardize MRI intensity values across different data sources.
- Each patient's modality is normalized independently by subtracting the mean and dividing by the standard deviation of the brain region.
- Images are clipped at [−5, 5] to remove outliers and rescaled to [0, 1], with the non-brain region set to 0.



