import torch, dataset, utils, modules, time, pickle
import matplotlib.pyplot as plt
from tqdm import tqdm

"""
Author name: Eli Cox
File name: train.py
Last modified: 22/11/2023
Create a new model or continue to train a previously created one.
This will be trained up to 100 epochs and validated every 5 epochs.
The process wil use stylegan to train a model to generate MRIs based on the OASIS dataset.
"""

def train_fn(
    critic,
    gen,
    mapping_network,
    path_length_penalty,
    loader,
    opt_critic,
    opt_gen,
    opt_mapping_network,
):
    # Set models to train
    gen.train()
    critic.train()
    mapping_network.train()
    # Generate fancy loader visualiser
    loop = tqdm(loader, leave=True)
    for batch_idx, real in enumerate(loop):
        # Load testing images
        real = real.to(utils.DEVICE)
        cur_batch_size = real.shape[0]

        # Generate noise inputs for generator
        w     = utils.get_w(mapping_network, cur_batch_size)
        noise = utils.get_noise(cur_batch_size)
        with torch.cuda.amp.autocast():
            # Generate fake images
            fake = gen(w, noise)
            
            # Critique fake images and real images
            critic_fake = critic(fake.detach())
            critic_real = critic(real)

            # Score overall model with adversarial loss function
            gp = utils.gradient_penalty(critic, real, fake, device=utils.DEVICE)
            loss_critic = (
                -(torch.mean(critic_real) - torch.mean(critic_fake))
                + utils.LAMBDA_GP * gp
                + (0.001 * torch.mean(critic_real ** 2))
            )

        # Backstep process
        critic.zero_grad()
        loss_critic.backward()
        opt_critic.step()

        gen_fake = critic(fake)
        loss_gen = -torch.mean(gen_fake)

        if batch_idx % 16 == 0:
            plp = path_length_penalty(w, fake)
            if not torch.isnan(plp):
                loss_gen = loss_gen + plp

        mapping_network.zero_grad()
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()
        opt_mapping_network.step()

        # Store first loss value for logging
        if batch_idx == 0:
            loss = loss_critic.item()

        loop.set_postfix(
            gp=gp.item(),
            loss_critic=loss_critic.item(),
        )
    return loss

def eval_fn(
    critic,
    gen,
    mapping_network,
    loader
):
    # Switch models to evaluation mode
    gen.eval()
    critic.eval()
    mapping_network.eval()
    total_loss = 0
    samples_validated = 0

    for real in loader:
        # Load validation images
        real = real.to(utils.DEVICE)
        cur_batch_size = real.shape[0]

        # Generate noise inputs for generator
        w     = utils.get_w(mapping_network, cur_batch_size)
        noise = utils.get_noise(cur_batch_size)
        with torch.cuda.amp.autocast():
            # Generate fake images
            fake = gen(w, noise)
            
            # Critique fake images and real images
            critic_fake = critic(fake.detach())
            critic_real = critic(real)

            # Score overall model with adversarial loss function
            gp = utils.gradient_penalty(critic, real, fake, device=utils.DEVICE)
            loss_critic = (
                -(torch.mean(critic_real) - torch.mean(critic_fake))
                + utils.LAMBDA_GP * gp
                + (0.001 * torch.mean(critic_real ** 2))
            )

        # Log loss function
        total_loss += loss_critic.item()
        samples_validated += 1
    return total_loss / samples_validated

# Loaders for training and validation image sets
loader_train        = dataset.create_data_loader("train")
loader_validate        = dataset.create_data_loader("validate")

# Individual models for StyleGAN
gen                 = modules.Generator(utils.LOG_RESOLUTION, utils.W_DIM).to(utils.DEVICE)
critic              = modules.Discriminator(utils.LOG_RESOLUTION).to(utils.DEVICE)
mapping_network     = modules.MappingNetwork(utils.Z_DIM, utils.W_DIM).to(utils.DEVICE)
path_length_penalty = modules.PathLengthPenalty(0.99).to(utils.DEVICE)

# Model optimisers
opt_gen             = torch.optim.Adam(gen.parameters(), lr=utils.LEARNING_RATE, betas=(0.0, 0.99))
opt_critic          = torch.optim.Adam(critic.parameters(), lr=utils.LEARNING_RATE, betas=(0.0, 0.99))
opt_mapping_network = torch.optim.Adam(mapping_network.parameters(), lr=utils.LEARNING_RATE, betas=(0.0, 0.99))

# Logging variables
losses = []
best_loss = float('inf')
validate_every_n_epochs = 5
start_time = time.time()
last_epoch = start_time

load_checkpoint = False
load_best = True

old_epoch = 0

starting_epoch = 1
if load_checkpoint:
    # Resume from previous training process
    start_time = 1697679243.0313046
    if load_best:
        gen.load_state_dict(torch.load(f"best_gen_{start_time}.pth"))
        critic.load_state_dict(torch.load(f'best_critic_{start_time}.pth'))
        mapping_network.load_state_dict(torch.load(f'best_map_{start_time}.pth'))
        with open(f'best_checkpoint_{start_time}.pickle', 'rb') as handle:
            starting_epoch, losses, best_loss = pickle.load(handle)
    else:
        gen.load_state_dict(torch.load(f"latest_gen_{start_time}.pth"))
        critic.load_state_dict(torch.load(f'latest_critic_{start_time}.pth'))
        mapping_network.load_state_dict(torch.load(f'latest_map_{start_time}.pth'))
        with open(f'latest_checkpoint_{start_time}.pickle', 'rb') as handle:
            starting_epoch, losses, best_loss = pickle.load(handle)
    with open(f'latest_checkpoint_{start_time}.pickle', 'rb') as handle:
        old_epoch, _, _ = pickle.load(handle)
    #start_time += 0.0000000001
    starting_epoch+=1

for epoch in tqdm(range(starting_epoch, utils.epochs + 1)):

    try:
        loss = train_fn(
            critic,
            gen,
            mapping_network,
            path_length_penalty,
            loader_train,
            opt_critic,
            opt_gen,
            opt_mapping_network,
        )
        # Save losses obtained from training the models
        losses.append(loss)

        print(f"Epoch {epoch:03d} | Training Loss: {loss}")
        if epoch % validate_every_n_epochs == 0:
            average_loss = eval_fn(
                critic,
                gen,
                mapping_network,
                loader_validate,
            )
            # Validate the model and generate training examples

            utils.generate_examples(mapping_network, gen, epoch, start_time)

            # Save model if better
            if average_loss < best_loss:
                best_loss = average_loss

                torch.save(gen.state_dict(), f'best_gen_{start_time}.pth')
                torch.save(critic.state_dict(), f'best_critic_{start_time}.pth')
                torch.save(mapping_network.state_dict(), f'best_map_{start_time}.pth')
                with open(f'best_checkpoint_{start_time}.pickle', 'wb') as handle:
                    pickle.dump([epoch, losses, best_loss], handle, protocol=pickle.HIGHEST_PROTOCOL)
            #save progress
            torch.save(gen.state_dict(), f'latest_gen_{start_time}.pth')
            torch.save(critic.state_dict(), f'latest_critic_{start_time}.pth')
            torch.save(mapping_network.state_dict(), f'latest_map_{start_time}.pth')
            with open(f'latest_checkpoint_{start_time}.pickle', 'wb') as handle:
                pickle.dump([epoch, losses, best_loss], handle, protocol=pickle.HIGHEST_PROTOCOL)

            # Plot the losses over number of epochs
            plt.plot(range(1, len(losses) + 1), losses, label='Training Loss')
            plt.title('Training Loss over Epochs')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig('loss_plot.png')
            plt.close()

            # Print validation results
            print(f"Epoch {epoch} | Validation Loss: {average_loss}")
        time_elapsed = time.time() - last_epoch
        print(f'Epoch {epoch} complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        #print(f"Time Running: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
        last_epoch = time.time()
    except KeyboardInterrupt:
        print ("\n\nUser cancelled training: saving progress")
        # Only save if loaded best and past latest, or loaded latest
        if (old_epoch < epoch) or (load_checkpoint == True and load_best == False):
            #save progress
            print ("Overriding last save")
            torch.save(gen.state_dict(), f'latest_gen_{start_time}.pth')
            torch.save(critic.state_dict(), f'latest_critic_{start_time}.pth')
            torch.save(mapping_network.state_dict(), f'latest_map_{start_time}.pth')
            with open(f'latest_checkpoint_{start_time}.pickle', 'wb') as handle:
                pickle.dump([epoch, losses, best_loss], handle, protocol=pickle.HIGHEST_PROTOCOL)
        break
