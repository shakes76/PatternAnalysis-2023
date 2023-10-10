# Classifier based on Siamese Network to Classify Alzheimer's Disease on ADNI Dataset

This repository implements a Siamese Network that will be trained to generate embeddings of  images of the ADNI dataset. A classifier (fully connected layers) will be added on top of the trained Siamese Network to classify Alzhemier's Disease on the dataset.

## General Overview of the Siamese Network
A Siamese Neural Network or twin neural network is an architecture that is made to differentiate between two inputs. It consists of two sub-networks that share the same weights and parameters. Each sub-network accept individual inputs, or in this case images, and produce feature vectors or embeddings. The euclidiean distance between the embeddings of both sub-networks are then calculated to determine the similarity between the inputs. The goal is to update the weights of the sub-networks so that the distance between inputs of the same class is small and the distance between inputs of different classes is large.

The loss function that is used in this implementation is the contrastive loss. The loss penalises large distances for similar pairs (pairs in different class) by increasing the loss as the distance grows. For dissimilar pairs (pairs in different classes), it penalises distances that are less than a specified margin, ensuring that images that are different should be at least a margin apart.

```
class ContrastiveLoss(torch.nn.Module):
   # label=0 for negative pairs, label=1 for positive pairs
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, distance, label):
        # Compute contrastive loss
        loss_contrastive = torch.mean((label) * torch.pow(distance, 2) +
                                      (1-label) * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2))
        return loss_contrastive
```

TBA graph
TBA explanation of the graph

## ADNI Dataset
The ADNI dataset consits of brain MRI images with the class labels AD (0) and NC(1). There are around the 

## Pre-processing

## Training

| Siamese Model Name | Epoch | Learning Rate | Margin | Classifier Name            | Epoch | Learning Rate | Test Result | Notes                                    |
| ------------------ | ----- | ------------- | ------ | -------------------------- | ----- | ------------- | ----------- | ---------------------------------------- |
| siamese_50.pth     | 50    | 0.1           | 1      | classifier_model_50_30.pth | 30    | 0.01          |             | pre-trained = True for resnet18 backbone |
| siamese_40.pth     | 40    | 0.01          | 1      | classifier_model_40_30.pth | 30    | 0.01          |             | pre-trained = True for resnet18 backbone |
## Testing

## Dependencies

## References