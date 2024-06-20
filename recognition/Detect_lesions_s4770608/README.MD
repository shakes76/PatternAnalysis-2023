## Customized Mask R-CNN for Skin Lesion Detection and Classification
### 1 Problem Statement
One of the most prevalent cancers in the world, skin cancer, can be efficiently treated if found early. Accurately identifying and categorizing skin lesions in dermoscopic pictures is a crucial first step in the diagnosis of skin cancer.

### 2 Algorithm
This project employs a customized Mask R-CNN model, tailored for the precise detection and classification of skin lesions as melanoma or seborrheic keratosis.





#### 2.1 Model Architecture
The customized Mask R-CNN uses a ResNet backbone for feature extraction. The model has been fine-tuned specifically for skin lesion detection tasks, with a particular emphasis on handling imbalanced datasets and enhancing the localization of the lesions with improved bounding boxes and masks. Custom anchors and a specialized loss function have been introduced to handle the unique challenges posed by skin lesion images.

The code of model is in modules.py. I will explain it detailly.

Loading the Pre-trained Model:
The function begins by loading a pre-trained Mask R-CNN model with a ResNet-50 backbone and FPN (Feature Pyramid Network). The version used here appears to be a custom or updated version, denoted by _v2.

```python
model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(pretrained=True)
```
Modifying the Classification Head:
To adapt the model for a different number of classes, the classifier head of the model is modified. The number of output features of the final classification layer is set equal to num_classes.

```python
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
```

Modifying the Mask Prediction Head:
Similarly, the mask prediction head is also customized. The hidden layer dimension is set to 256, and the output features are set to num_classes to match the new dataset.

```python
in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)
```

Parameter Tuning Setup:
All parameters in the model are set to be trainable. It means that during the fine-tuning process, the weights of all layers can be updated.In the original setting only, last three layer are update.
```python
in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
for name, para in model.named_parameters():
    para.requires_grad = True
```




#### 2.1 Data preprocessing
Training Transformations for Images and Masks: 

(1)Random Vertical Flip:With a probability of 0.3, the images and masks are flipped vertically.
v2.RandomVerticalFlip(p=0.3)

(2)Random Horizontal Flip:With a probability of 0.3, the images and masks are flipped horizontally.
v2.RandomHorizontalFlip(p=0.3)

(3)Random Rotation:The images and masks are rotated randomly within a range of 0 to 180 degrees.
v2.RandomRotation(degrees=(0,180))


(4)Random Resized Crop:A random size and aspect ratio is selected to crop the images and masks, and the crops are resized to a size of 1129x1504 pixels.
v2.RandomResizedCrop(size=(1129, 1504))

(5)Conversion to Tensor:The images and masks are converted to tensors, which are suitable for model input.
v2.ToTensor()

(6)Normalization for Training Images: train_transform_stage2_for_img
The images are normalized using the mean and standard deviation values from the ImageNet dataset.
v2.Normalize(mean=imagenet_mean, std=imagenet_std)

In validation stage, only Normalize is applied



#### 2.3 Data spliting
The ISIC2017 dataset  are splitted already to traning, valid, testing 

### 3  Implementation Detail

#### 3.1 Custom dataset class.
Because Maskrcnn model requires bounding boxes and images as input, however , ISIC dataset are images and masks. Thus, I need convert mask to bounding boxes in custom the dataset class.

Here is a detailed explanation 
Overview:
CustomISICDataset is a specialized dataset class tailored for processing and loading skin lesion images and their corresponding masks for object detection tasks. The dataset is primarily designed to handle RGB images as inputs and binary masks, which delineate the regions of interest.

Initialization:
Upon initialization, the class requires paths to the CSV file containing labels, image directory, and mask directory. It also accepts optional transformations and a target size for the images and masks.

csv_file: Path to the CSV file containing image labels.
img_dir: Directory containing the image files.
mask_dir: Directory containing the mask files.
transform_stage1_for_img_mask: Initial transformations applied to both images and masks.
transform_stage2_for_img: Additional transformations applied primarily to images.
Integrity Checks:
The _check_dataset_integrity function ensures that the dataset's integrity is maintained by confirming the correspondence between image IDs and mask IDs, ensuring there are no mismatches or missing files.

Data Loading:
The __getitem__ method loads an individual image-mask pair based on the provided index, applying the necessary transformations and generating a target dictionary containing bounding boxes, labels, and masks.

Image and Mask Loading:

Images are loaded and converted to RGB format.
Masks are loaded as single-channel images.
Transformations:

Specified transformations are applied sequentially. A loop ensures that the mask retains essential information post-transformation.
Label Processing:

Labels are derived based on the conditions specified, ensuring a unique label for each state.
Bounding Box Calculation:

Bounding boxes are determined based on the masks.

A target dictionary is formed containing the bounding boxes, labels, and masks, ensuring a structured format for model training.




Requirements and Dependencies
Python 3.9

torch                       2.0.0+cu118

torchvision                 0.15.1+cu118

shapely                     2.0.2

tqdm                        4.65.0


To ensure reproducibility, a requirements.txt file is provided. 
Install the necessary packages using the following command:


pip install -r requirements.txt
#### 3.1 Training
modified the dataset path in get_data_loaders function in main.py then you are good to go
```python
def get_data_loaders(target_size):
```
```console
python main.py
```

#### 3.1 predict
modified the dataset path in following code snippiet  in predict.py then you are good to go. The output images will be saved to argument output_path directory
```python
test_data = CustomISICDataset \
    (csv_file='/home/ubuntu/works/code/working_proj/2023S3Course/COMP3710/project/data/test_label.csv',
     img_dir='/home/ubuntu/works/code/working_proj/2023S3Course/COMP3710/project/data/test_imgs',
     mask_dir='/home/ubuntu/works/code/working_proj/2023S3Course/COMP3710/project/data/test_gt',
     transform_stage1_for_img_mask=val_transform_stage1_for_img_mask,
     transform_stage2_for_img=val_transform_stage2_for_img,
     target_size=target_size)

```
```console
python predict.py  --output_path output
```



###  Result

#### Metrics used, IOU and Average precision
The Inter-section Over Union between bouding boxes is used. The Average precision is used for classification. 
#### Visulization 
A detailed visualizaiton could be cound at  https://wandb.ai/baibizhe-s-team/ISIC?workspace=user-baibizhe

Good result:

![good1.png](good1.png)
![good2.png](good2.png)


Bad result:
![bad1.png](bad1.png)
![bad2.png](bad2.png)
#### Quantitative 
The final IOU I've reached is 0.724 and mean accuracy is 0.513 
