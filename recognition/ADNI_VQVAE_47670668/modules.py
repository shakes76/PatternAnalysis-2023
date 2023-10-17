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
    def __init__(self, embedding_dim, num_embeddings, commitment_cost, dtype=torch.float32):
        super(VectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        embedding_shape = (embedding_dim, num_embeddings)
        initializer = nn.init.uniform_
        self.embeddings = nn.Parameter(initializer(torch.empty(embedding_shape, dtype=dtype)))

    def forward(self, inputs):
        flat_inputs = inputs.contiguous().view(-1, self.embedding_dim)

        distances = (
            torch.sum(flat_inputs**2, dim=1, keepdim=True) -
            2 * torch.matmul(flat_inputs, self.embeddings) +
            torch.sum(self.embeddings**2, dim=0, keepdim=True)
        )

        encoding_indices = torch.argmax(-distances, dim=1)
        encoding_indices = encoding_indices.view(*inputs.shape[:-1])
        encodings = F.one_hot(encoding_indices, num_classes=self.num_embeddings).to(distances.dtype)

        # Reshape encoding_indices to match the shape of inputs
        encoding_indices = encoding_indices.view(inputs.shape[:-1] + (-1,))

        quantized = self.quantize(encoding_indices)

        # print("quantized from self.quantized:", quantized.shape)

        e_latent_loss = torch.mean((quantized.detach() - inputs)**2)
        q_latent_loss = torch.mean((quantized - inputs.detach())**2)
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()
        quantized = quantized.view(*inputs.shape)
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return {
            'quantize': quantized,
            'loss': loss,
            'perplexity': perplexity,
            'encodings': encodings,
            'encoding_indices': encoding_indices,
            'distances': distances,
        }

    def quantize(self, encoding_indices):
        quantized = self.embeddings.t()[encoding_indices].contiguous()
        quantized = quantized.squeeze(3)
        # print("quantized after squeeze", quantized.shape)
        return quantized


class VQVAEModel(nn.Module):
    def __init__(self, encoder, decoder, vqvae, pre_vq_conv1, data_variance):
        super(VQVAEModel, self).__init__()
        self._encoder = encoder
        self._decoder = decoder
        self._vqvae = vqvae
        self._pre_vq_conv1 = pre_vq_conv1
        self._data_variance = data_variance

    def forward(self, inputs):
        z = self._pre_vq_conv1(self._encoder(inputs))
        # print("output after encoder and _pre_vq_conv1", z.shape)

        z = z.permute(0,2,3,1)
        # print("output after permute", z.shape)

        vq_output = self._vqvae(z)  # Unpack the tuple

        quantize = vq_output['quantize'].permute(0, 3, 1, 2)
        x_recon = self._decoder(quantize)
        recon_error = torch.mean((x_recon - inputs) ** 2) / self._data_variance
        total_loss = recon_error + vq_output['loss']  # Use the 'loss' from the VQ-VAE module

        return {
            'z': z,
            'x_recon': x_recon,
            'loss': total_loss,
            'recon_error': recon_error,
            'vq_output': vq_output
        }


class VQVAEModel(nn.Module):
    def __init__(self, encoder, decoder, vqvae, pre_vq_conv1, data_variance):
        super(VQVAEModel, self).__init__()
        self._encoder = encoder
        self._decoder = decoder
        self._vqvae = vqvae
        self._pre_vq_conv1 = pre_vq_conv1
        self._data_variance = data_variance

    def forward(self, inputs):
        z = self._pre_vq_conv1(self._encoder(inputs))
        # print("output after encoder and _pre_vq_conv1", z.shape)

        z = z.permute(0,2,3,1)
        # print("output after permute", z.shape)

        vq_output = self._vqvae(z)  # Unpack the tuple

        quantize = vq_output['quantize'].permute(0, 3, 1, 2)
        x_recon = self._decoder(quantize)
        recon_error = torch.mean((x_recon - inputs) ** 2) / self._data_variance
        total_loss = recon_error + vq_output['loss']  # Use the 'loss' from the VQ-VAE module

        return {
            'z': z,
            'x_recon': x_recon,
            'loss': total_loss,
            'recon_error': recon_error,
            'vq_output': vq_output
        }
