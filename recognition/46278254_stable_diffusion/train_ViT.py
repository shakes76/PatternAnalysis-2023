# ==== import from package ==== #
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from einops import rearrange, repeat
from collections import defaultdict
from torch.utils.data import TensorDataset, DataLoader
# ==== import from this folder ==== #
from model_VAE import VQVAE
from util import compact_large_image, sinusoidal_embedding
DEVICE = torch.device("cuda")
print("DEVICE:", DEVICE)

mode = 'VQVAE'
load_epoch = 45

# Get latent set
latent_set = torch.load(f'collected_latents/{mode}_{load_epoch}.pt')
vae = torch.load(f'model_ckpt/{mode}/epoch_AE_{load_epoch}.pt')
vae.eval()

# Collected indices data
with torch.no_grad():
    latents, z_indices = latent_set.tensors[0], latent_set.tensors[1]
    quant, diff_loss, ind = vae.quantize(latents)
    ind = rearrange(ind, '(b 1 h w) -> b (h w)', b=len(latents),
                    h=vae.z_shape[0], w=vae.z_shape[1])
    dataset = TensorDataset(ind, z_indices)
    dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True)
    print(ind.shape)


class PixelCNN(nn.Module):

    def __init__(self, vae):
        super(PixelCNN, self).__init__()

        # Define main embedding.
        self.embed_dim = 32
        self.n_embed = vae.quantize.n_e
        self.z_shape = vae.z_shape
        self.embed = nn.Embedding(vae.quantize.n_e, self.embed_dim)

        # Define Transformer Decoder.
        # Note that We don't have use Transformer Encoder because there's no sequential condition information.
        # And we just use the output of mlp(condition's information) as decoder's input (or, memory)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.embed_dim, nhead=2)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)

        # Define Positional Encoding. (This position encoding is 0-th pixel ~ 255-th pixel)
        self.positional_encoding = sinusoidal_embedding(
            self.z_shape[0] * self.z_shape[1], self.embed_dim)

        # Project Decoder output to classification (predict which indices we use.)
        self.predictor = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.SiLU(),
            nn.Linear(self.embed_dim, self.n_embed),
        )

        # Conditional MLP. output embedding information after giving z-index of brain
        self.cond_mlp = nn.Sequential(
            nn.Embedding(32, self.embed_dim),
            nn.SiLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )

    def forward(self, dst, z_idx):
        N, S = dst.shape

        # Change indices to embedding and add positional encoding
        h = self.embed(dst) + \
            self.positional_encoding[None, :, :].to(dst.device)

        # Generate the decoder's input / memory.
        memory = self.cond_mlp(z_idx)

        # Transpose to correct input (Decoder eat Sequence, Batch_size, Embedding_size)
        # Various sequence has same memory so we repeat S times.
        memory = repeat(memory, 'N E -> S N E', S=S)

        # Add gaussian noise to give some uncertainty.
        memory = memory + torch.randn_like(memory)

        # Generate Decoder's mask. (the output of index 1 shouldn't peep output of after index 1)
        mask = nn.Transformer.generate_square_subsequent_mask(S).to(DEVICE)

        # Transpose to correct input (Decoder eat Sequence, Batch_size, Embedding_size)
        h = rearrange(h, 'N S E -> S N E')

        # Predict the output
        h = self.decoder(h, memory, tgt_mask=mask)
        h = self.predictor(h)

        # Transpose to batch_first form
        h = rearrange(h, 'S N E-> N S E')

        return h

    def sample(self, batch_size):
        # T: z-index conditions
        T, N, S, E = 32, batch_size, 16*16, self.embed_dim

        # Generate random memory with condition.
        # This part is same as forward but repeat T times.

        # Condition embedding
        z_idx = torch.arange(0, 32, step=1, device=DEVICE)
        cond = self.cond_mlp(z_idx)
        memory = torch.randn([N, T, E]).to(DEVICE)
        memory = memory + cond[None, :, :]
        memory = repeat(memory, 'N T E -> S (N T) E', S=S)

        # Get positional embedding and decoder mask
        pos_enc = self.positional_encoding[None, :, :].to(DEVICE)
        mask = nn.Transformer.generate_square_subsequent_mask(S).to(DEVICE)

        # Repeatedly generate output indices.
        gen_ind = torch.zeros(S, N * T).long().to(DEVICE)
        for i in tqdm(range(S), total=S):

            # Transpose to batch_first=False
            rev_ind = rearrange(gen_ind, 'S (N T) -> (N T) S', N=N, T=T)
            h = self.embed(rev_ind) + pos_enc
            h = rearrange(h, '(N T) S E -> S (N T) E', N=N, T=T)

            # Pass decoder & MLP
            h = self.decoder(h, memory, tgt_mask=mask)
            h = self.predictor(h)

            # Transpose to batch_first=True
            h = rearrange(h, 'S (N T) E-> (N T) S E', N=N, T=T)

            # Get current output (i-th pixel indice)
            h = h[:, i, :].detach()

            # Use weighted random sampler to predict next pixel.
            # I use random sampler instead of torch.argmax() beacuse
            # I want to add some uncertainty to avoid mode collapse.
            from torch.utils.data import WeightedRandomSampler
            cur_gen_ind = torch.zeros([N * T, ]).long().to(DEVICE)
            for idx in range(N * T):
                # It seems like randomsmapler cannot support gpu tensor.
                # Thus we detach the statistics.
                weight = h[idx].detach().cpu()
                # Error will occur if the weight is all zero so I use softmax before weight sampling
                weight = torch.nn.functional.softmax(weight+1e-6, dim=0)
                sample = WeightedRandomSampler(weight, num_samples=1)
                # We only get one sample as our output.
                sample = list(sample)[0]
                cur_gen_ind[idx] = sample

            # If you don't want to use random sampler, just do the following line
            # cur_gen_ind = torch.argmax(h, dim=1)
            gen_ind[i] = cur_gen_ind

        # Transpose to batch_first=True
        return rearrange(gen_ind, 'S (N T) -> N T S', N=N, T=T)


# Define model & criteria & optim
pixelCNN = PixelCNN(vae).to(DEVICE)
criteria = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(pixelCNN.parameters(), lr=2e-4)


def train():
    total_loss = 0
    for now_step, batch_data in tqdm(enumerate(dataloader), total=len(dataloader)):
        ind, z_idx = [data.to(DEVICE) for data in batch_data]

        optimizer.zero_grad()
        out = pixelCNN(ind.detach(), z_idx.detach())

        # Change 3-d shape into 2-d shap becasue CE only support 2-d tensor
        out = rearrange(out, "N S E -> (N S) E")
        ind = rearrange(ind, "N S -> (N S)")
        loss = criteria(out, ind)

        # Update network weights
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)


# Train 5 epochs is enough.
for i in range(5):
    print(f"epoch {i}: loss: {train(i)}")
    torch.save(pixelCNN, f'model_ckpt/visualTransformer/model_{i}.pt')

# Testing
with torch.no_grad():
    # We sample 6 images for visualization
    for sample_idx in range(6):
        # Get one random sample
        out_inds = pixelCNN.sample(1)[0]
        # Pass embedding layer after generating indices.
        z_q = vae.quantize.embedding(out_inds)
        # reshape to 3-d
        z_q = rearrange(z_q, 'b (h w) c -> b c h w', w=16)
        # Generate z-index condition
        cond = torch.arange(0, 32, 1).long().to(DEVICE)
        # Generate image from VAE
        sample = vae.decode(z_q, cond).cpu()

        # Visualize the output
        from util import compact_large_image
        imgs = compact_large_image(sample, HZ=4, WZ=8)
        for idx in range(imgs.shape[0]):
            plt.imsave(
                f'visualize/Transformer_vis/{sample_idx}.png', imgs[idx] * 0.5 + 0.5, cmap='gray')
