# Classifying ADNI Brain Data for Alzheimer's Using Vision Transformer

Alzheimer's disease is a brain condition that causes memory loss and makes it hard for people to think and carry out daily tasks, getting worse over time. With the advancement of deep learning technology, it is now possible to use deep learning to detect Alzheimer's disease from MRI scans. This project uses the Vision Transformer model to classify MRI brain scans as either Normal Cognitive (NC) or Alzheimer's disease (AD) and aims to achieve an accurayc of above 80%.

## Architecture
The Vision Transformer uses tranformers,originally used for for natural language processing tasks, on image data. It works by first dividing the images into patches, then embeding them as sequences and finally processing them with transformer blocks.

![ViT architecture](https://github.com/bquek00/PatternAnalysis-2023/blob/2c189675d69af3c897474e3076d9c15dc9fa83dd/recognition/DontForgetAlzheimers/Screenshot%202023-10-26%20at%203.52.47%20AM.png)

The provided code in this repository uses the ViT architecture to process images. It first divides input images into 16x16 patches, then embeds them into vectors. They are then processed by the transformer encoder which uses a multi-head self-attention mechanism and then a feed-foward neural network for each embedding. 

## Preproccessing
Data  preprocessing was completed in dataset.py. It uses the ADNI dataset, provided on the COMP3710 blackboard, which contains:

- 11120 NC for training
- 10400 AD for training
- 4540 NC for test
- 4460 AD for test

Images are first resized to (224 x 224) then a RandomHorizontalFlip is applied. Images are also normalised to have a mean of 0.5 and a standard deviation of 0.5 to help with stability and convergence while training.
  

## Training
Training of the model is done in train.py. The model was trained over 20 epochs. The hyperparameters were chosen due to convention and also from published research. Below are the hyperparameters:

- Patch Size: 16x16
- Learning Rate: 0.001
- Layers: 12
- Transformer Architecture: 8 heads

These were chosen from prior research from the ViT paper by Google Research and the "Attention is All You Need" paper. For example, the original ViT paper uses the 16x16 patches as thi gave a good balance between granularity and representational capacity [1]. Also, Vaswani proposed a multi-headed attention with 8 heads in their paper {2}. 

## Testing

Testing code is also located in train.py

After 20 epochs, the testing gave a result of 73.29%. This was based off the accuracy evaluation metric which is calculated by 

- $(TP + TN) / (TP + TN + FP + FN)$

![Loss over 20 epochs](https://github.com/bquek00/PatternAnalysis-2023/blob/8f5cdea5170649a1be91abc45fab251eafcb0843/recognition/DontForgetAlzheimers/LOSS.png)
  
## Dependencies 

- Pillow==9.4.0
- Pillow==10.1.0
- torch==2.0.1
- torchvision==0.15.2

## Install and Run

- Download all dependencies
- Clone the repo git ```git clone https://github.com/bquek00/PatternAnalysis-2023.git```
- Navigate to DontForgetAlzheimers directory ```cd PatternAnalysis-2023/recognition/DontForgetAlzheimers```
- Check out topic-recognition brnach ```git checkout topic-recognition```
- Run the predict.py for usage ```python3 predict.py```
- To test the model accuracy on the whole dataset, run train.py ```python3 train.py```


