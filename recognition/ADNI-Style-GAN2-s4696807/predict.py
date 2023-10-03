from train import * 

from torch import optim
from torchvision.utils import save_image
import os

def generate_examples(gen, epoch, n=100):
    
    gen.eval()
    for i in range(n):
        with torch.no_grad():
            w     = get_w(1)
            noise = get_noise(1)
            img = gen(w, noise)
            if not os.path.exists(f'saved_examples/epoch{epoch}'):
                os.makedirs(f'saved_examples/epoch{epoch}')
            save_image(img*0.5+0.5, f"saved_examples/epoch{epoch}/img_{i}.png")

    gen.train()
    
gen.train()
critic.train()
mapping_network.train()

        
loader              = get_loader(DATASET, LOG_RESOLUTION, BATCH_SIZE)
path_length_penalty = PathLengthPenalty(0.99).to(DEVICE)
opt_gen             = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.99))
opt_critic          = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.99))
opt_mapping_network = optim.Adam(mapping_network.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.99))

for epoch in range(EPOCHS):
    train_fn(
        critic,
        gen,
        path_length_penalty,
        loader,
        opt_critic,
        opt_gen,
        opt_mapping_network,
    )
    if epoch % 50 == 0:
    	generate_examples(gen, epoch)
