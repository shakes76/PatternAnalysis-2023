# --------------------------------------------------------------------------------
# File: modules.py
# Author: Indira Devi Rusvandy
# Date: 2023-10-20
# Description: 
#   This script implements a Vector Quantized Variational Autoencoder (VQ-VAE) using PyTorch.
#   The implementation includes the definition of the Encoder, Decoder, VectorQuantizer, 
#   ResidualStack, and the complete VQVAEModel. The VQ-VAE is a generative model that 
#   quantizes the representations of the input data, which can be beneficial for various 
#   tasks like image generation, compression, etc.
#
#   The Encoder module compresses the input into a lower-dimensional space, the VectorQuantizer 
#   performs the quantization of these representations, and the Decoder module reconstructs 
#   the input from these quantized representations. The ResidualStack is used within the Encoder 
#   and Decoder to aid in learning complex representations.
#
# Usage:
#   This script is designed to be used as a module in machine learning projects where VQ-VAEs are
#   applicable. Instantiate the VQVAEModel with the desired configuration and use it in your training.
#
#   Example:
#       encoder = Encoder(...)
#       decoder = Decoder(...)
#       vq = VectorQuantizer(...)
#       vqvae_model = VQVAEModel(encoder, decoder, vq, ...)
#       # Train the model
# --------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualStack(nn.Module):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(ResidualStack, self).__init__()
        self._num_hiddens = num_hiddens
        self._num_residual_layers = num_residual_layers
        self._num_residual_hiddens = num_residual_hiddens
        self._relu = nn.ReLU()

        self._layers = nn.ModuleList()  # Use nn.ModuleList to register the layers
        for _ in range(num_residual_layers):
            conv3 = nn.Conv2d(
                in_channels=self._num_hiddens,
                out_channels=num_residual_hiddens,
                kernel_size=3,
                stride=1,
                padding=1)  # Added padding to keep spatial dimensions consistent
            conv1 = nn.Conv2d(
                in_channels=num_residual_hiddens,
                out_channels=num_hiddens,
                kernel_size=1,
                stride=1)
            self._layers.append(nn.Sequential(conv3, conv1))

    def forward(self, inputs):
      h = inputs
      for layer in self._layers:
          conv3_out = layer[0](self._relu(h))
          conv1_out = layer[1](self._relu(conv3_out))
          h = h + conv1_out
      return self._relu(h)



class Encoder(nn.Module):
  def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens):
    super(Encoder, self).__init__()
    self._num_hiddens = num_hiddens
    self._num_residual_layers = num_residual_layers
    self._num_residual_hiddens = num_residual_hiddens
    self._relu = nn.ReLU()

    self._enc_1 = nn.Conv2d(
        in_channels=1,
        out_channels=self._num_hiddens // 2,
        kernel_size=(4, 4),
        stride=(2, 2),
        padding=1)  
    self._enc_2 = nn.Conv2d(
        in_channels=self._num_hiddens // 2,
        out_channels=self._num_hiddens,
        kernel_size=(3, 3),   # Changed kernel size from 4x4 to 3x3
        stride=(2, 2),
        padding=1)  
    self._enc_3 = nn.Conv2d(
        in_channels=self._num_hiddens,
        out_channels=self._num_hiddens,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=1)  
    self._residual_stack = ResidualStack(
        self._num_hiddens,
        self._num_residual_layers,
        self._num_residual_hiddens)

  def forward(self, x):
    # print("input:", x.shape)
    h1 = self._relu(self._enc_1(x))
    h2 = self._relu(self._enc_2(h1))
    h3 = self._relu(self._enc_3(h2))
    return self._residual_stack(h3)


class Decoder(nn.Module):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens, input_channels):
        super(Decoder, self).__init__()
        self._num_hiddens = num_hiddens
        self._num_residual_layers = num_residual_layers
        self._num_residual_hiddens = num_residual_hiddens
        self._relu = nn.ReLU()

        self._dec_1 = nn.Conv2d(
            in_channels=input_channels,
            out_channels=self._num_hiddens,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=1)
        self._residual_stack = ResidualStack(
            self._num_hiddens,
            self._num_residual_layers,
            self._num_residual_hiddens)

        self._dec_2_pre = nn.ConvTranspose2d(
            in_channels=self._num_hiddens,
            out_channels=self._num_hiddens,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=1)

        self._dec_2 = nn.ConvTranspose2d(
            in_channels=self._num_hiddens,
            out_channels=self._num_hiddens // 2,
            kernel_size=(4, 4),
            stride=(2, 2),
            padding=1)

        self._dec_3 = nn.ConvTranspose2d(
            in_channels=self._num_hiddens // 2,
            out_channels=1,
            kernel_size=(4, 4),
            stride=(2, 2),
            padding=1)

    def forward(self, x):
      h1 = self._dec_1(x)
      h1 = self._relu(h1)
      h2 = self._residual_stack(h1)
      h2 = self._dec_2_pre(h2)
      h3 = self._relu(h2)
      h3 = self._dec_2(h3)
      h4 = self._relu(h3)
      x_recon = self._dec_3(h4)
      return x_recon

    

class VectorQuantizer(nn.Module):
    def __init__(self, embedding_dim, num_embeddings, commitment_cost):
        super(VectorQuantizer, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        # Initialize the embedding space
        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embeddings.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)

    def forward(self, inputs):
      
        # Flatten input
        flat_input = inputs.reshape(-1, self.embedding_dim)

        # Calculate distances
        distances = (
            torch.sum(flat_input**2, dim=1, keepdim=True)
            - 2 * torch.matmul(flat_input, self.embeddings.weight.t())
            + torch.sum(self.embeddings.weight**2, dim=1)
        )

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings).to(inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Restore spatial dimensions for encoding_indices
        encoding_indices = encoding_indices.view(inputs.shape[:-1])

        # Quantize
        quantized = self.embeddings(encoding_indices).squeeze(1)
        quantized = quantized.view(inputs.shape)

        # Use the Straight Through Estimator for the gradients in the backward pass
        quantized = inputs + (quantized - inputs).detach()

        # Perplexity
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return {
            'quantize': quantized,
            'perplexity': perplexity,
            'encodings': encodings,
            'encoding_indices': encoding_indices,
            'distances': distances,
        }

class VQVAEModel(nn.Module):
    def __init__(self, encoder, decoder, vq, pre_vq_conv1):
        super(VQVAEModel, self).__init__()
        self._encoder = encoder
        self._decoder = decoder
        self._vq = vq
        self._pre_vq_conv1 = pre_vq_conv1

    def forward(self, inputs):
        z = self._encoder(inputs)
        z = self._pre_vq_conv1(z).permute(0, 2, 3, 1)
        vq_output = self._vq(z)
        x_recon = self._decoder(vq_output['quantize'].permute(0, 3, 1, 2))

        return {
            'z': z,
            'x_recon': x_recon,
            'vq_output': vq_output
        }
    