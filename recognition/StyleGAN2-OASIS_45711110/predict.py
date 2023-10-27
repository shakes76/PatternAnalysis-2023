"example usage of the trained model"

import torch
import os
from torchvision.utils import save_image

from utils import get_noise, get_w

def generate_examples(gen, epoch, n=100):
    
    gen.eval()
    alpha = 1.0
    for i in range(n):
        with torch.no_grad():
            w     = get_w(1)
            noise = get_noise(1)
            img = gen(w, noise)
            if not os.path.exists(f'saved_examples/epoch{epoch}'):
                os.makedirs(f'saved_examples/epoch{epoch}')
            save_image(img*0.5+0.5, f"saved_examples/epoch{epoch}/img_{i}.png")

    gen.train()