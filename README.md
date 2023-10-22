
# DDPM with U-Net for Image Denoising

This repository contains an implementation of a U-Net architecture augmented with time embeddings, designed to work with the Diffusion Denoising Probabilistic Model (DDPM).


## Author

**Name**: Santiago Rodrigues  
**Student Number**: 46423232  
**Course**: COMP3710

This work was undertaken as part of the COMP3710 course.

## Key Components

#### Dataset

- **`ADNIDataset`**: A PyTorch dataset class to handle data from the AD and NC folders for both training and testing. It assumes data is split into two directories: `train` and `test`, each containing the classes `AD` and `NC`.
- **`get_data_loader`**: A utility function to get a DataLoader for the dataset.

#### Modules

1. **Block**: A basic building block for the network. It contains two convolutional layers interspersed with SiLU activations and has Layer Normalization.
2. **UNet**: A modified U-Net architecture with time embeddings:
   - Time embeddings are sinusoidal embeddings that provide information about the diffusion timestep to the network.
   - The encoder and decoder segments of the U-Net have been modified to use the `Block` defined above.
   - Each segment of the U-Net (encoder and decoder) has time embeddings that are combined with feature maps at different scales.
3. **DDPM_UNet**: A wrapper around the U-Net for the DDPM model:
   - Accepts original images and introduces noise based on a provided timestep.
   - Uses the U-Net architecture to estimate the denoised version of a noisy image for a given timestep.

#### Training 

**File**: `train.py`

1. **Dataset Loading & Preprocessing**:
    - The dataset is resized to `224x224` and normalized to `[-1,1]`.
    - DataLoader is created with a batch size of `32` and shuffle enabled.
  
2. **Model Initialization**:
    - The `DDPM_UNet` is initialized with given `min_beta`, `max_beta`, and `num_steps`.

3. **Training Loop**:
    - The training loop runs for `30` epochs. 
    - MSE Loss between the estimated noise and the actual noise introduced during the forward pass is computed.
    - If the loss is the best so far, the model is saved.
  
4. **Hyperparameters**:
    - Learning Rate: `0.001`
    - Number of Steps: `1000`
    - Min Beta: `1e-3`
    - Max Beta: `0.03`

### Usage

1. **Dataset Loading**:
    ```python
    dataloader = get_data_loader(root_dir=path_to_data)
    ```

2. **Model Initialization**:
    ```python
    model = DDPM_UNet(UNet(num_steps=NUM_STEPS), num_steps=NUM_STEPS, min_beta=MIN_BETA, max_beta=MAX_BETA, device=DEVICE)
    ```

3. **Denoising**:
    ```python
    noisy_image = model(original_image, time_step)
    denoised_image = model.denoise(noisy_image, time_step)
    ```

4. **Training**:
    To train the model, run the `train.py` script. The model's weights will be saved if the epoch loss improves.
    ```python
    training_loop(model, loader, NUM_EPOCHS, optimizer, DEVICE)
    ```

## Image Generation and GIF Creation with DDPM U-Net

The `predict.py` script provides functionality for generating new images using the trained DDPM U-Net model and creating a GIF showcasing the denoising process over time.

### Key Features

1. **`create_gif`**: Generates a GIF from the trained model, visualizing the denoising process.
    - **Arguments**:
        - `model`: The trained DDPM U-Net model.
        - `num_samples`: Number of images to generate. Default is `16`.
        - `device`: CUDA device or CPU for computation.
        - `frames_per_gif`: Number of frames in the GIF. Default is `100`.
        - `gif_name`: Name of the output GIF. Default is `"sampling.gif"`.
        - `c, h, w`: Channels, height, and width of the images. Defaults are `1, 224, 224`.

2. **`get_frame_indices`**: Determines the indices for frames in the GIF.

3. **`process_noise_with_model`**: Processes the noise tensor with the model for a single timestep.

4. **`create_frame`**: Normalizes and arranges images into a frame for the GIF.

5. **`store_gif`**: Stores the frames as a GIF.

### Usage

1. **Generating a GIF**:
    ```python
    create_gif(model_trained, num_samples=16, device=my_device, frames_per_gif=100, gif_name="sampling.gif")
    ```
    This will generate a GIF named `sampling.gif` visualizing the denoising process of the model over time.

---

## Training Process of the DDPM U-Net Model

Training the model involved playing with the hyperparameters, until a suitable results were achieved. Due to Google Collab not saving a few figures, I only have the batch size comparison.

The loss gives insight into how well the model is approximating the target function. Lower loss indicates that the model's predictions are closer to the actual values.

### Training Details:

1. **Dataset**: The model was trained on the ADNIDataset with images resized to `224x224` pixels.

2. **Optimization Details**: A learning rate of `0.001` was used. The training loop includes gradient clipping and an optional learning rate scheduler.

3. **Hardware**: Training was done on a machine with GPU support. If unavailable, the CPU was used as a fallback.

### Performance Metrics:

To evaluate the model's training performance, we have visualized the loss metrics with respect to different batch sizes:

1. **Average Loss per Epoch for Batch Size 32**:
    ![Average Loss per Epoch for Batch Size 32](figures/batch_size_32.png)

2. **Average Loss per Epoch for Batch Size 128**:
    ![Average Loss per Epoch for Batch Size 128](figures/batch_size_128.png)

3. **Loss per Step for Both Batch Sizes**:
    While the average loss per epoch provides a macroscopic view, the loss per step offers a granular perspective of the model's learning behavior within each epoch.

### Visualization of Model's Image Generation:

To get a visual understanding of the model's performance, a GIF showcasing the image generation over time was created:

![DDPM U-Net Image Generation Over Time](figures/ddpm_generated_samples.gif)

This GIF visually represents the model's ability to generate and denoise images throughout its training lifecycle.

## Dependencies

To ensure consistent behavior across different setups, it's recommended to use the provided `environment.yml` file with Miniconda. This will create a new conda environment with all necessary dependencies installed.

1. **Install Miniconda or Anaconda**: Download Miniconda from: [Miniconda's official page](https://docs.conda.io/en/latest/miniconda.html) 

2. **Create a new Conda environment**:
    Navigate to the project directory where the `environment.yml` file is located and run the following command:
    ```bash
    conda env create -f environment.yml
    ```

3. **Activate the new environment**:
    ```bash
    conda activate comp3710
    ```

## References

Implementation adapted from the following articles. They also delves deeper into the theoretical foundation and practical implementation of DDPMs:

1. **Generating Images with DDPMs: A PyTorch Implementation**:
    - Source: [Medium Article](https://medium.com/mlearning-ai/enerating-images-with-ddpms-a-pytorch-implementation-cef5a2ba8cb1)
    - This article provides an overview of generating images using DDPMs and offers a comprehensive PyTorch implementation guide.

2. **Annotated Diffusion**:
    - Source: [HuggingFace Blog](https://huggingface.co/blog/annotated-diffusion)
    - A detailed exposition of the diffusion process, this article by HuggingFace demystifies the underlying concepts using annotations.

3. **Attention Is All You Need**:
    - Authors: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin
    - Year: 2023
    - [arXiv Link](https://arxiv.org/abs/1706.03762)

4. **Denoising Diffusion Probabilistic Models**:
    - Authors: Jonathan Ho, Ajay Jain, Pieter Abbeel
    - Year: 2020
    - [arXiv Link](https://arxiv.org/abs/2006.11239)
