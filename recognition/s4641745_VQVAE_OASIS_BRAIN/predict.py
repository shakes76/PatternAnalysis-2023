import os
import torch as t

from dataset import MODEL_PATH
from utils import save_image

device = t.device('cuda' if t.cuda.is_available() else 'cpu')
if not t.cuda.is_available():
    print("Warning CUDA not found. Using CPU")

vqvae = t.load(os.path.join(MODEL_PATH, 'vqvae.txt'))
G = t.load(os.path.join(MODEL_PATH, 'generator.txt'))

x = t.randn(1, 100, 1, 1).to(device)
with t.no_grad():
    f1 = G(x)
gen_codebook_idx = f1[0][0]
save_image(gen_codebook_idx, 'generated-codebook-final.png')

# save generated embeddings
gen_codebook_idx = t.flatten(gen_codebook_idx)
unique_vals = [134, 418]
input_min = t.min(gen_codebook_idx)
input_max = t.max(gen_codebook_idx)
num_intervals = len(unique_vals)
interval_size = (input_max - input_min) / num_intervals

for i in range(0, num_intervals):
    MIN = input_min + i * interval_size
    gen_codebook_idx[
        t.logical_and(
            MIN <= gen_codebook_idx,
            gen_codebook_idx <= (MIN + interval_size))] = unique_vals[i]

gen_embedding = gen_codebook_idx.view(64, 64)
save_image(gen_embedding, 'generated-embedding-single-sample.png')

gen_codebook_idx = gen_codebook_idx.long()

z_q = vqvae.vq.quantize(gen_codebook_idx)
decoded = vqvae.decoder(z_q)
test_quantized = decoded[0][0]
save_image(test_quantized, 'generated-decoded-final.png')