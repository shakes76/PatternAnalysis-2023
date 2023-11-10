## Lesion detection and classification using YOLOV7 on the ISIC2017 dataset

### Table of Contents
- [Lesion detection and classification using YOLOV7 on the ISIC2017 dataset](#lesion-detection-and-classification-using-yolov7-on-the-isic2017-dataset)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
- [Dataset](#dataset)
  - [Overview](#overview)
  - [Preprocessing](#preprocessing)
  - [Usage](#usage)
  - [Model Architecture: open source YOLOV7 Model](#model-architecture-open-source-yolov7-model)
    - [Core ideas used in the YOLOV1 paper:](#core-ideas-used-in-the-yolov1-paper)
  - [Output](#output)
  - [Training](#training)
    - [Training discussion](#training-discussion)
  - [Results](#results)
    - [Confusion Matrix](#confusion-matrix)
    - [F1 - curve](#f1---curve)
    - [Precision - Recall curve](#precision---recall-curve)
  - [Results discussion](#results-discussion)
    - [Downsampling](#downsampling)
    - [Anchor boxes misaligned](#anchor-boxes-misaligned)
  - [Example outputs](#example-outputs)


### Installation
- Prerequisites: python=3.10.12 && cuda=11.7

**A GPU cluster is used for this project, more specifically rangpur @ UQ. Therefore a lot of the training and inference scripts are based on slurm jobs. If needed this can easily be converted to run locally.**

```
git clone git@github.com:larsmoan/PatternAnalysis-2023.git
git submodule init 
git submodule update
pip install -r requirements.txt
```

## Dataset
**Source**: [ISIC 2017 Dataset](https://challenge.isic-archive.com/data/#2017)

### Overview
Each image comes with corresponding label and segmentation file highlighting the lesion.
- **Training Set**: 
  - 2000 images.
  
- **Validation Set**: 
  - 600 images.
  
- **Test Set**: 
  - 150 images.

**Lesion Classes**:
- `Melanoma`
- `Seborrheic Keratosis`
- `Nevi / Uknown`: Technically known as a benign skin lesion. Commonly referred to as a mole.


### Preprocessing
Given that the dataset provides segmentation files, there's a need for preprocessing to convert these labels into YOLO bounding box labels. 

Steps include:
1. Identify the maximum and minimum coordinates within the segmentation area.
2. Fit a bounding box around this region.
3. Assign the class based on the label provided in the associated CSV file.

More information can be found in the file: [dataset_utils.py](./dataset_utils.py)

The dataset itself also needs to be refactored a bit to work with YOLOV7, therefore the structure is changed to the following:
```
dataset/
│
├── train/
│   ├── img_1.jpg
│   ├── ...
│   ├── img_n.jpg
│   ├── img_1.txt
│   ├── ...
│   └── img_n.txt
│
├── val/
│   ├── img_1.jpg
│   ├── ...
│   ├── img_n.jpg
│   ├── img_1.txt
│   ├── ...
│   └── img_n.txt
│
└── test/
    ├── img_1.jpg
    ├── ...
    ├── img_n.jpg
    ├── img_1.txt
    ├── ...
    └── img_n.txt
```

The prepocessed dataset can be downloaded from this link:
https://drive.google.com/uc?id=1YI3pwanX35i7NCIxKnfXBozXiyQZcGbL or from [dataset_utils.py](./dataset_utils.py)



### Usage
- Download the dataset and pretrained yolov7 weights:
  ```
  python dataset_utils.py
  ```
- Train the model:
  Using rangpur cluster:
  ```
  sbatch run_custom_train.sh
  ```
  Or using Google Colab:
  [isic_train.ipynb](./isic_train.ipynb)
- Run inference on testset:
  ```
  sbatch run_test.sh
  ```

### Model Architecture: open source [YOLOV7 Model](https://github.com/WongKinYiu/yolov7)

YOLOV7 is based on the original YOLO paper: [YOLOV1](https://arxiv.org/abs/1506.02640) which was published in 2015 and presented a leap in inference speed for object detection models. The main reason for this was that it was one of the first models that did object detection in a single stage, hence the name YOLO ( you only look once ) in contrast to the two stage models that were popular at the time. Note that some single stage models were present, such as SSD, but they had relatively poor accuracy performance.

#### Core ideas used in the YOLOV1 paper:

The original paper was trained on input images of size 448x448 and these images where parsed into a grid of 7x7 grid cells.
<figure style="margin-right: 10px; display: inline-block;">
   <img src="./figures/grid_cell_yolo.png" alt="Example Image" width="460" height="340">
   <figcaption>Original grid cells on 448x448 image. Source: <a href="http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Redmon_You_Only_Look_CVPR_2016_paper.pdf">"You Only Look Once by Joseph Redmon et al., CVPR 2016</a>.</figcaption>
</figure>

The idea was that each grid cell was responsible for predicting a object if the center of that object was within the given grid cell.
**In this example the red grid cell would be responsible to detecting the car present in the top right corner**

 In the paper it was also proposed that each grid cell's output was two bounding boxes each with their own confidence / objectness score + a class probability vector. I.e was each grid cell only capable of predicting one detection, altough it could predict two bounding boxes. The bounding box with the highest objectness score was chosen in addition to the class with the highest probability.

<figure style="margin-right: 10px; display: inline-block;">
   <img src="./figures/yolov1.png" alt="Example Image" width="710" height="340">
  <figcaption>YOLOv1 output tensor. Source: <a href="http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Redmon_You_Only_Look_CVPR_2016_paper.pdf">"You Only Look Once by Joseph Redmon et al., CVPR 2016</a>.</figcaption>
</figure>

In the figure above the core ideas is presented, each grid cell proposing two bounding boxes (in black) but only one class. The "best" box and class over a certain treshold is then used as the final prediction. -> Single stage object detection and classification!

### Output
Because the model uses a 7x7 grid, the output is a tensor of shape (batch_size, 7, 7, 30). Where 30 is the number of parameters that is predicted for each grid cell. The 30 parameters per grid cell are:
[objectness_score, box_x, box_y, width, height] + [objectness_score, box_x, box_y, width, height] + [class_probabilities].
Two bounding boxes + a vector of class probabilities. In the case of the original YOLO paper, the model was trained on the [PASCAL dataset](http://host.robots.ox.ac.uk/pascal/VOC/) where there are 20 classes present. Hence the 30 parameters per grid cell.


<figure style="margin-right: 10px; display: inline-block;">
   <img src="./figures/output-tensor-yolov1.png.webp" alt="YOLOv1 Output Tensor Visualization" width="580" height="340">
   <figcaption>YOLOv1 output tensor. Source: <a href="http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Redmon_You_Only_Look_CVPR_2016_paper.pdf">"You Only Look Once by Joseph Redmon et al., CVPR 2016</a>.</figcaption>
</figure>


### Training
Training was done on the rangpur cluster using a variety of different GPU's, the two main training runs shown in the plots below was done using the A100 and the P100. (A100 for the batch size of 8 and P100 for the batch size of 32).

- Hyperparameters that was used is described in: [hyp.scratch.p6.yaml](./hyp.scratch.p6.yaml)

***In the plots below, the term step refers to epoch number***

<figure style="margin-right: 10px; display: inline-block;">
   <img src="./results/train-boxloss.png" alt="Example Image" width="600" height="300">
  <figcaption>Box loss.</figcaption>
</figure>
<figure style="margin-right: 10px; display: inline-block;">
   <img src="./results/train-clsloss.png" alt="Example Image" width="600" height="300">
  <figcaption>Class loss.</figcaption>
</figure>
<figure style="margin-right: 10px; display: inline-block;">
   <img src="./results/train-objloss.png" alt="Example Image" width="600" height="300">
  <figcaption>Object loss.</figcaption>
</figure>


#### Training discussion
From the plots showing the box, class and object loss it is clear that they indeed was decreasing when the training was halted. With that said they decreased by small amounts, almost negligible. In the case of the A100 training run, that took about 48 hours.

An interesting observation is the fact that during both runs the model's loss reduced drastically for the first 10 or so epochs, after this the reduction was almost 0. This can be an indication that given the data the models were presented they actually fitted really quickly, but that the data itself might have very complex structures. Hence the model was not able to learn more from the data.


### Results
<figure style="margin-right: 10px; display: inline-block;">
   <img src="./results/metrics-mAP0.5_0.95.png" alt="Example Image" width="600" height="300">
  <figcaption>mAP 0.5:0.95.</figcaption>
</figure>
<figure style="margin-right: 10px; display: inline-block;">
   <img src="./results/metrics-mAP_0.5.png" alt="Example Image" width="600" height="300">
  <figcaption>mAP@0.5.</figcaption>
</figure>
<figure style="margin-right: 10px; display: inline-block;">
   <img src="./results/metrics-precision.png" alt="Example Image" width="600" height="300">
  <figcaption>Precision</figcaption>
</figure>
<figure style="margin-right: 10px; display: inline-block;">
   <img src="./results/metrics-recall.png" alt="Example Image" width="600" height="300">
  <figcaption>Recall</figcaption>
</figure>


#### Confusion Matrix
<img src="results/yolov7_b32_p100/confusion_matrix.png" alt="Description" >


#### F1 - curve
<img src="results/yolov7_b32_p100/F1_curve.png" alt="Description">

#### Precision - Recall curve
<img src="results/yolov7_b32_p100/PR_curve.png" alt="Description">

### Results discussion
The results are not very good, and far off what I anticipated when first embarking on this project. The best result was an mAP@0.5 of about 0.718 which is quite poor.

An important observation is the instability in training related to precision and recall, we can see that for thease metrics the models doesnt improve much past the 10 first epochs, but it has huge variety from epoch to epoch.

Initially I thought these issues were related to the small batch size, that was the main driver for increasing it to 32 during the second training run.
However this did not improve the results drastically, although it had a significant effect on mAP@0.5.

I think the reason for the results being as poor as they are is the inherent complexity of the dataset, when I have went through the data personally I often find it really challenging to understand the labels in the ISIC2017 dataset. 

The label unknown / benign is especially tricky to understand as a human because there are a lot of artifacts present in the dataset that is not labeled as benign skin lesions.

#### Downsampling
The original dataset too large to fit on a student node at rangpur, since the storage limit there is 16gb. Therefore I dowsampled the dataset by a factor of 2 and used this during both training and testing. Originally I thought this wouldn't pose any problem as the original dataset has images of really high resolution, often in the range og 5MB per image.
That being said, the downsamplign is in itself a loss of information as the dimensionality is reduced, so it could be a factor that has contributed to the poor results.

#### Anchor boxes misaligned
After diving deeper into the intrinsics of the YOLO architecture I have found one major factor which I suspect could enhance the performance by a ton. Using another set of anchor boxes!
The standard anchor boxes utilized by YOLOV7 does not fit that well with the labels present in the ISIC dataset, where we can see that often a bounding box takes up almost the complete image.

To mitigate this issue I should have performed a custom annchor box analyzis, by doing a clustering on the boxes present in the ISIC dataset and therefore "allow" the model to do bigger predictions. This is something I will try to implement in the future.

### Example outputs
<figure style="margin-right: 10px; display: inline-block;">
   <img src="./results/yolov7_b32_testing8/test_batch0_labels.jpg" alt="Example Image" width="600" height="350">
  <figcaption>Batch 0 - Labels</figcaption>
</figure>
<figure style="margin-right: 10px; display: inline-block;">
   <img src="./results/yolov7_b32_testing8/test_batch0_pred.jpg" alt="Example Image" width="600" height="350">
  <figcaption>Batch 0 - Predictions</figcaption>
</figure>

<figure style="margin-right: 10px; display: inline-block;">
   <img src="./results/yolov7_b32_testing8/test_batch1_labels.jpg" alt="Example Image" width="600" height="350">
  <figcaption>Batch 1 - Labels</figcaption>
</figure>
<figure style="margin-right: 10px; display: inline-block;">
   <img src="./results/yolov7_b32_testing8/test_batch1_pred.jpg" alt="Example Image" width="600" height="350">
  <figcaption>Batch 1 - Predictions</figcaption>
</figure>


In these outputs what I discussed related to anchor boxes above is clear, a lot of the labels take up almost all of the image. This is quite far off from the YOLO standard anchor boxes.

Another way of visualizing this is by the plot below, here we can clearly see an example of how the anchor boxes are misaligned with the labels present in the ISIC dataset.
<figure style="margin-right: 10px; display: inline-block;">
   <img src="./figures/anchor_comparison.png" alt="Example Image" width="600" height="350">
  <figcaption>Anchor comparison</figcaption>
</figure>
