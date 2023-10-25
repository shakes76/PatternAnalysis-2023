# Classifying ADNI Brain Data for Alzheimer's Using Vision Transformer

Alzheimer's disease is a brain condition that causes memory loss and makes it hard for people to think and carry out daily tasks, getting worse over time. With the advancement of deep learning technology, it is now possible to use deep learning to detect Alzheimer's disease from MRI scans. This project uses the Vision Transformer model to classify MRI brain scans as either Cognitively Normal (CN) or Alzheimer's disease (AD) and aims to achieve an accurayc of above 80%.

## Architecture
The Vision Transformer uses tranformers,originally used for for natural language processing tasks, on image data. It works by first dividing the images into patches, then embeding them as sequences and finally processing them with transformer blocks.
![ViT architecture](Screenshot 2023-10-26 at 3.52.47 AM.png)

The provided code in this repository uses the ViT architecture to process images. It first divides input images into 16x16 patches, then embeds them into vectors. They are then processed by the transformer encoder which uses a multi-head self-attention mechanism and then a feed-foward neural network for each embedding. 

## Preproccessing

## Design choices

## Testing

## Example prediction
