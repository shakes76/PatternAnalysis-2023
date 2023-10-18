# Diffusion Model on the OASIS Brain Dataset

## Project Overview
In this project, a generative model based on the Diffusion algorithm has been developed using PyTorch to simulate the [OASIS Brain Dataset](https://www.oasis-brains.org/). The objective is to generate images of sufficiently high clarity that closely resemble the anatomical structures found in the original dataset.

<details open>
  <summary>Table of Contents</summary>

- [Dependencies](#dependencies)
- [File Structure](#file-structure)
- [Data Loading and Preprocessing](#data-loading-and-preprocessing)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Usage](#usage)
- [References](#references)

  ### Some Javascript
  ```js
  function logSomething(something) {
    console.log('Something', something);
  }
  ```
</details>

## Dependencies
```python
python - 3.11.4
pytorch - 2.0.1
torchvision - 0.15.2
matplotlib - 3.8.0
```

## File Structure
`imports.py` Imports all the required libraries and modules.

`dataset.py` Contains the ImageDataset class for data loading and process_dataset function for preprocessing.

`modules.py` Defines the forward pass of the diffusion network, consisiting of encoder and decoder blocks made of ResNet layers.

`train.py` Sets up and trains a diffusion model and plots the training loss over iterations.

`predict.py` Visualise the reverse diffusion process using a pre-trained diffusion model.

`utils.py` Defines essential components for implementing both the forward and backward dynamics of a generative diffusion model.

## Data Loading and Preprocessing
In this project, the objective is to generate high-quality brain images that closely resemble those found in the OASIS brain dataset. To optimize the model's performance, both training and test datasets have been merged, thereby increasing the overall number of images available for training.

```python
train_data = ImageDataset(
    directory=train_dir, image_transforms=image_transforms)
test_data = ImageDataset(
    directory=test_dir, image_transforms=image_transforms)

# Combine training and test datasets into single dataset for training
combined_data = ConcatDataset([train_data, test_data])
```

Though numerical validation metrics were not a requirement, the architecture has been designed to accommodate future validation efforts. Metrics such as the Structural Similarity Index [(SSIM)](https://github.com/VainF/pytorch-msssim) or Fréchet Inception Distance [(FID)](https://github.com/mseitzer/pytorch-fid) could be integrated in subsequent development phases for more rigorous evaluation.

## Model Architecture
Diffusion models are a class of generative model aims to model  the underlying probability distribution of data—in this case, 2D images. The training procedure involves starting with actual data samples and transforming them into noise through a sequence of diffusion steps. The model learns to execute a series of denoising operations that can reverse this process, thereby generating new samples from the learned data distribution during inference.

The architecture of our model is inspired by the U-Net framework and consists of a series of EncoderBlocks and DecoderBlocks. Each of these blocks is constructed using ResNet units as the fundamental computational elements.

![Model Architecture](images/architecture.png)
The architectural design of this project is informed by the research presented in the paper "High-Resolution Image Synthesis with Latent Diffusion Models" by Robin Rombach et al., accessible [here](https://arxiv.org/pdf/2112.10752.pdf). It should be noted that the conditional aspects of the model discussed in the paper are not implemented in this project, as the OASIS brain dataset does not contain conditional variables.

To incorporate temporal information into the model, we have integrated Sinusoidal Position Embeddings. This design choice is informed by the research presented in the paper "Denoising Diffusion Probabilistic Models" by Jonathan Ho et al., which can be accessed [here](https://arxiv.org/pdf/2006.11239.pdf).


#### Model Flow:
The architecture of the U-Net model implemented is inspired by the seminal work "U-Net: Convolutional Networks for Biomedical Image Segmentation" by Olaf Ronneberger et al.. The paper can be accessed through [here](https://arxiv.org/pdf/1505.04597.pdf).

![U-Net Architecture](images/unet.png)

1. **Encoding Path**: Four EncoderBlocks (`down1` to `down4`) progressively downsample the input while capturing spatial features. Intermediate `skip` outputs are stored.
   
2. **Bottleneck**: A standalone ResNet block (`bottle_neck`) serves as the bottleneck layer with a higher channel dimension.
   
3. **Decoding Path**: Four DecoderBlocks (`up1` to `up4`) upsample the bottleneck output, utilizing 'skip' outputs from the encoding path for feature fusion.
  
4. **Output Layer**: The final output is normalized with BatchNorm and passed through a 1x1 convolution to produce the model's output.

## Training
The Adam optimizer is selected as the optimization algorithm, and the loss function employed is the Huber loss, calculated between the true and predicted noise values.
```python
optimizer = optim.Adam(model.parameters(), lr=1e-3)
```
```python
loss = F.smooth_l1_loss(noise, predicted_noise)
```

![Training Loss](/images/training_loss.png)

## Usage
1. In the `dataset.py` file, adapt the `process_dataset` function to partition the data into training, testing, and validation sets. To verify that the data has been successfully loaded, uncomment and execute the code in the `main` block.

```python
def process_dataset(batch_size=8, is_validation=False,
                    train_dir="WRITE ME", 
                    test_dir="WRITE ME", 
                    val_dir="WRITE ME"):
```

2. Execute the `train.py` script to train and save the model. You can customize the training process by adjusting parameters such as `epochs` and `batch_size`. Additionally, specify the target directory for storing the generated plot by modifying the appropriate variable at the end of the script. Upon successful execution, the output should resemble the example provided [here](#training).

```python
save_dir = os.path.expanduser("WRITE ME")
```

3. Execute the `predict.py` script to generate synthetic brain images from pure noise. To specify a different directory for storing the generated images, modify the designated section at the end of the script.

```python
 save_dir = os.path.expanduser("WRITE ME")
```

<br>
The following sample output was generated after training the model for 51 epochs on the OASIS Brain dataset. The reasonable clarity and fidelity of these images serve as a testament to the model's performance capabilities.

![](/images/result_process.png)
![](/images/image_grid.png)

## References
- [OASIS Brain Dataset](https://www.oasis-brains.org/)
- [(SSIM)](https://github.com/VainF/pytorch-msssim)
- [(FID)](https://github.com/mseitzer/pytorch-fid)
- [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/pdf/2112.10752.pdf)
- [Denoising Diffusion Probabilistic Model](https://arxiv.org/pdf/2006.11239.pdf)
- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/pdf/1505.04597.pdf)