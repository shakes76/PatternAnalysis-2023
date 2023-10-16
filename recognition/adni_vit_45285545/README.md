# Vision Transformer for Alzheimer's Disease Classification

This project applies the Vision Transformer architecture to the task of classifying MRI brain images as either Cognitive Normal (CN) or representative of Alzheimer's disease (AD). The ViT is trained and tested on patient-level splits of the dataset from the [Alzheimer's Disease Neuroimaging Initiative (ADNI)](http://adni.loni.usc.edu). A test accuracy of 75.56% is achieved.

## Model Architecture

The Vision Transformer (ViT) is a self-attention-based architecture which has consistently demonstrated state-of-the-art performance on various image recognition benchmarks while requiring substantially fewer computational resources to train than convolutional neural networks (CNNs), which are now superseded as the ubiquitous architecture for computer vision tasks [1].

ViTs extend the Transformer architecture introduced by Vaswani et al. [4] for natural language processing tasks [1]. The following figure presents a graphic representation of the model architecture from Dosovitsky et al. [1].

![Vision Transformer architecture from [2]](static/vit-model-architecture.png)

Transformers rely on attention, which calculates the pairwise inner product for every pair of tokens in a set [2]. This method has less inductive bias than other architectures, enabling better generalisation and hence better performance given enough training data [2]. However, as a quadratic operation, attention is too expensive to apply to each individual pixel in an image [2]. Images are therefore divided into patches, which are vectorised and linearly projected using an embedding matrix [2]. The projected patches are combined with positional embeddings, which inform the relative positions of the patches [2]. These _embedded patches_ can then be input into a regular Transformer. An extra learnable class embedding is input alongside the embedded patches, whose trained output feeds into the final fully-connected classification layers to yield a prediction.

## Data and Preprocessing

### Dataset

The ViT is trained and tested on patient-level splits of the dataset from the [Alzheimer's Disease Neuroimaging Initiative (ADNI)](http://adni.loni.usc.edu). The train-test split dataset was obtained from the COMP3710 course page on the UQ Blackboard website.

The training split contains 21,520 images:
- 11,120 images are cognitive normal (NC); and
- 10,400 images are Alzheimer's disease (AD).

The test split contains 9000 images:
- 4540 images are cognitive normal (NC); and
- 4460 images are Alzheimer's disease (AD).

All subsets contain exactly 20 images per patient and there are no common patients between the sets. The training set is additionally 80/20 patient-level split into training and validation sets. Again, there are no common patients between the sets.

### Preprocessing

Preprocessing of the data consists of data augmentation and type conversion. In order, images are subject to:

1. A [`RandomHorizontalFlip`](https://pytorch.org/vision/0.15/generated/torchvision.transforms.RandomHorizontalFlip.html).

2. A [`RandomCrop`](https://pytorch.org/vision/main/generated/torchvision.transforms.RandomCrop.html) to 224x224, with padding size of 4 and `reflect` padding mode.

3. Conversion to tensor float type.

The above transforms are only all applied to training images. Validation and testing images are only subject to type conversion.

## Model Training and Results

This solution uses the [`vit_b_16`](https://pytorch.org/vision/main/models/generated/torchvision.models.vit_b_16.html) model provided by PyTorch. The classification head is replaced with a fully-connected layer with 2 outputs rather than 10. The solution leverages transfer learning using the [`IMAGENET1K_V1`](https://pytorch.org/vision/main/models/generated/torchvision.models.vit_b_16.html) weights, also from PyTorch. Transfer learning avoids retraining the more general, lower-level feature extractors of the ViT, which have been trained to high efficiency on the larger ImageNet dataset. Only the higher-level feature extractors and fully-connected classification head need to be retrained to be specific to the ADNI dataset.

The model is trained for 16 epochs or until validation loss does not improve for five consecutive epochs, at which point early stopping triggers and the model is reverted to the state following the epoch with the lowest validation loss. The training also uses the [`ReduceLROnPlateau`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html) scheduler to decrease the learning rate by a factor of 10 if the validation loss does not improve for two consecutive epochs.

The following figure presents example model training and validation metrics.

![Sample training metrics](static/adni-vit-metrics-1697362188.png)

The model is evaluated on the testing set by running inference on all testing images and recording the model predictions. Since there are exactly 20 samples of every patient, the predictions for each patient are tallied and the more frequent prediction is used as the overall prediction for the patient.

The model trained above achieves the following result on the test split of the ADNI dataset.

<!-- ```
100%|███████████████████████████| 141/141 [01:53<00:00,  1.24it/s]
Test accuracy: 75.56% (00:01:53.86790)
``` -->

<!-- TODO: add image results; test image and prediction -->

## Reproducing Results

### Dependencies

This implementation uses Python 3.10 and PyTorch 2.1.0+cu118. PyTorch (specifically `torch` and `torchvision`) can be installed for the machine of choice following: https://pytorch.org/get-started/locally/

Other Python package dependencies include: `pandas`, `tqdm`, `gradio` and `seaborn`. These can be installed all at once using:
```
pip install pandas tqdm gradio seaborn
```
Of these packages, `seaborn` is only required for generating plots of training metrics from generated CSV files. If this is not a forseen use case, installing `seaborn` may be omitted and can always be installed later if found necessary.

A virtual environment ([`venv`](https://docs.python.org/3/library/venv.html)) is recommended for the installation, but is not mandatory.

### Setup and Run

1. Navigate to `recognition/adni_vit_45285545`.

2. Run the following to train the model and print a test result. The `--pg` option will show training progress bars. The model is automatically saved to file after training.
```
python train.py N_EPOCHS [--pg]
```

3. Run the following with a saved model file to run inference. The `--test` option will run the model on the test split of the ADNI dataset and print a test result. The `--gui` option will start a web-based GUI to interactively use the model.
```
python predict.py MDLFILE [--test] [--gui]
```

_NOTE: while the GUI accepts any image, using samples from the ADNI test set are useful because then the correct labels are known._

## References

[1] A. Dosovitsky et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale", arXiv: 2010.11929 [cs.CV], 2021.

[2] Yannic Kilcher. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (Paper Explained). (Oct. 4, 2020). Accessed: Oct. 11, 2023. [Online video]. Available: https://www.youtube.com/watch?v=TrdevFK_am4

[3] K. S. Krishnan and K. S. Krishnan, "Vision Transformer based COVID-19 Detection using Chest X-rays," 2021 6th International Conference on Signal Processing, Computing and Control (ISPCC), Solan, India, 2021, pp. 644-648, doi: 10.1109/ISPCC53510.2021.9609375.

[4] A. Vaswani et al., "Attention Is All You Need", arXiv: 1706.03762 [cs.CL], 2017.
