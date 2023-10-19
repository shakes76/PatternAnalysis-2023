import os

import torch as t
import torch.nn as nn
import torch.utils.data
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from dataset import vqvae_test_loader, vqvae_train_loader, MODEL_PATH, transform, GanDataset
from models import VQVAE, Generator, Discriminator
from utils import save_image

device = t.device('cuda' if t.cuda.is_available() else 'cpu')
if not t.cuda.is_available():
    print("Warning CUDA not found. Using CPU")

# ========== TRAIN VQVAE ==========

# VQVAE Hyper params
LR_VQVAE = 1e-3
BATCH_SIZE_VQVAE = 32
MAX_EPOCHS_VQVAE = 4
NUM_HIDDENS = 128
RESIDUAL_INTER = 32
NUM_EMBEDDINGS = 512
EMBEDDING_DIM = 64
BETA = 0.25
DATA_VARIANCE = 0.0338
LOG_STEP = 100

# # create VQVAE model
# model = VQVAE(NUM_HIDDENS, RESIDUAL_INTER, NUM_EMBEDDINGS, EMBEDDING_DIM, BETA)
# model.to(device)
#
# # Init optimizer
# optimizer = torch.optim.Adam(model.parameters(), lr=LR_VQVAE)
# train_recon_loss = []
#
# # train VQVAE
# print("VQVAE Training started")
# for i in range(MAX_EPOCHS_VQVAE):
#     print(f"EPOCH [{i+1}/{MAX_EPOCHS_VQVAE}]")
#
#     size = len(vqvae_train_loader.dataset)
#     batch_losses = []
#     i = 0
#     for batch, (X, _) in enumerate(vqvae_train_loader):
#         X = X.to(device)
#
#         optimizer.zero_grad()
#         vq_loss, data_recon = model(X)
#
#         recon_error = F.mse_loss(data_recon, X) / DATA_VARIANCE
#         loss = recon_error + vq_loss
#         loss.backward()
#         optimizer.step()
#         batch_losses.append(recon_error.item())
#
#         if (i+1) % LOG_STEP == 0:
#             print(f"Step {i+1} -  recon_error: {np.mean(batch_losses[-100:])}")
#         i += 1
#
#     loss = sum(batch_losses) / len(batch_losses)
#
#     train_recon_loss.append(loss)
#     print(f"Reconstruction loss: {loss}")
#
# # Save model
# t.save(model, os.path.join(MODEL_PATH, "vqvae.txt"))

model = t.load(os.path.join(MODEL_PATH, 'vqvae.txt'))

# save samples of real and test data
real_imgs1 = next(iter(vqvae_test_loader)) # load some from test dl
real1 = real_imgs1[0]
real1 = real1.to(device)
_, test_recon = model.forward(real1) # forward pass through vqvae to create reconstruction
save_image(real1, 'real-sample.png')
save_image(test_recon, 'recon-sample.png')

# visualise embeddings and quantized outputs
real_imgs2 = next(iter(vqvae_test_loader)) # load some from test dl
real2 = real_imgs2[0][0].unsqueeze(0)
real2 = real2.to(device)
encoded = model.encoder(real2)
conv = model.conv1(encoded)
_, _, _, codebook_indices = model.vq(conv)
test_codebook = codebook_indices.view(64, 64).float()
z_q = model.vq.quantize(codebook_indices)
decoded = model.decoder(z_q)
test_quantized = decoded[0][1]
save_image(real2, 'real-single-sample.png')
save_image(test_codebook, 'codebook-single-sample.png')
save_image(test_quantized, 'quantized-single-sample.png')


# ========== TRAIN GAN ==========

# GAN Hyper params
LR_GAN = 4e-4
BATCH_SIZE_GAN = 256
MAX_EPOCHS_GAN = 20
Z_DIM_GAN = 100
SAMPLE_NUM_GAN = 32
LOG_STEP_GAN = 10

# define dataset
gan_train_ds = GanDataset(model, transform)
gan_train_dl = t.utils.data.DataLoader(gan_train_ds, batch_size=BATCH_SIZE_GAN)

# define models
G = Generator()
D = Discriminator()
G = G.to(device)
D = D.to(device)

# criterion
g_optimizer = torch.optim.Adam(G.parameters(), lr=LR_GAN, betas=(0.5, 0.999))
d_optimizer = torch.optim.Adam(D.parameters(), lr=LR_GAN, betas=(0.5, 0.999))
criterion = nn.BCELoss().to(device)

# train GAN
# taken from COMP3710 Lab Demo 2 Part 3 (GAN) by Luke Halberstadt
fixed_z = Variable(torch.randn(SAMPLE_NUM_GAN, Z_DIM_GAN)).to(device)
print("GAN Training started")
for epoch in range(MAX_EPOCHS_GAN):
    for i, (images, _) in enumerate(gan_train_dl):
        batch_size = images.shape[0]
        # Build mini-batch dataset
        image = Variable(images).to(device)
        # Create the labels which are later used as input for the BCE loss
        real_labels = Variable(torch.ones(batch_size)).to(device)
        fake_labels = Variable(torch.zeros(batch_size)).to(device)

        # train discriminator
        outputs = D(image)
        d_loss_real = criterion(outputs, real_labels)  # BCE
        real_score = outputs

        # compute loss using fake images
        z = Variable(torch.randn(batch_size, Z_DIM_GAN, 1, 1)).to(device)
        fake_images = G(z)
        outputs = D(fake_images)
        d_loss_fake = criterion(outputs, fake_labels)  # BCE
        fake_score = outputs

        # Backwards propagation + optimize
        d_loss = d_loss_real + d_loss_fake
        D.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        # train generator
        z = Variable(torch.randn(batch_size, Z_DIM_GAN, 1, 1)).to(device)
        fake_images = G(z)
        outputs = D(fake_images)

        # We train G to maximize log(D(G(z))) instead of minimizing log(1-D(G(z)))
        # For the reason, see the last paragraph of section 3. https://arxiv.org/pdf/1406.2661.pdf
        g_loss = criterion(outputs, real_labels)  # BCE

        # Backprob + Optimize
        D.zero_grad()
        G.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        if (i + 1) % LOG_STEP_GAN == 0:
            # print("Epoch [%d/%d], Step[%d/%d], d_loss: %.4f, g_loss: %.4f, D(x): %.2f, D(G(z)): %.2f" % (
            #     epoch, max_epoch, i + 1, total_batch, d_loss.data.item(), g_loss.data.item(), real_score.data.mean(),
            #     fake_score.data.mean()))
            print("Epoch [%d/%d], Step[%d/%d], D(x): %.2f, D(G(z)): %.2f" % (
                epoch, MAX_EPOCHS_GAN, i + 1, len(gan_train_dl.dataset) // batch_size, real_score.data.mean(),
                fake_score.data.mean()))

# save models
t.save(G, os.path.join(MODEL_PATH, "generator.txt"))
t.save(D, os.path.join(MODEL_PATH, "discriminator.txt"))

# save generated codebook indices
x = torch.randn(1, 100, 1, 1).to(device)
with torch.no_grad():
    f1 = G(x)
gen_codebook_idx = f1[0][0]
save_image(gen_codebook_idx, 'generated-codebook-single-sample.png')

# save generated embeddings
gen_codebook_idx = torch.flatten(gen_codebook_idx)
unique_vals = [134, 418]
input_min = torch.min(gen_codebook_idx)
input_max = torch.max(gen_codebook_idx)
num_intervals = len(unique_vals)
interval_size = (input_max - input_min) / num_intervals

for i in range(0, num_intervals):
    MIN = input_min + i * interval_size
    gen_codebook_idx[
        torch.logical_and(
            MIN <= gen_codebook_idx,
            gen_codebook_idx <= (MIN + interval_size))] = unique_vals[i]

gen_embedding = gen_codebook_idx.view(64, 64)
save_image(gen_embedding, 'generated-embedding-single-sample.png')

gen_codebook_idx = gen_codebook_idx.long()
gen_output = model.vq.quantize(gen_codebook_idx)
gen_output = model.decoder(gen_output)
gen_decoded = gen_output[0][0]
save_image(gen_decoded, 'generated-decoded-single-sample.png')


