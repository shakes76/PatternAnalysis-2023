# Vision Transformer for Alzheimer's Disease Classification

This project applies the Vision Transformer architecture to the task of classifying MRI brain images as either Cognitive Normal (CN) or representative of Alzheimer's disease (AD). The ViT is trained and tested on patient-level splits of the dataset from the [Alzheimer's Disease Neuroimaging Initiative (ADNI)](http://adni.loni.usc.edu). A test accuracy of X% is achieved.

## Model Architecture

The Vision Transformer (ViT) is a self-attention-based architecture which has consistently demonstrated state-of-the-art performance on various image recognition benchmarks while requiring substantially fewer computational resources to train than convolutional neural networks (CNNs), which are now superseded as the ubiquitous architecture for computer vision tasks [1].

## Model Results

## Reproducing Results

### Hardware

These results were generated on a platform with:

- An NVIDIA A100 GPU with 10GB of memory
- An AMD EPYC 7542 CPU with 32 cores

The script `goslurm_COMP3710Project_RangpurTrain` was used to batch the training task.

### Dependencies

This implementation uses Python 3.10 and package dependencies are listed in `requirements.txt`.

### Setup and Run

1. Navigate to `recognition/adni_vit_45285545`.

2. Execute the following to train the model then save the trained model to the working directory:
```bash
python train.py N_EPOCHS
```

3. Execute the following with a saved model file to run inference using the test split of the ADNI dataset:
```bash
python predict.py MDLFILE
```

## References

[1] A. Dosovitsky et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale", arXiv: 2010.11929 [cs.CV], 2021.

[2] K. S. Krishnan and K. S. Krishnan, "Vision Transformer based COVID-19 Detection using Chest X-rays," 2021 6th International Conference on Signal Processing, Computing and Control (ISPCC), Solan, India, 2021, pp. 644-648, doi: 10.1109/ISPCC53510.2021.9609375.
