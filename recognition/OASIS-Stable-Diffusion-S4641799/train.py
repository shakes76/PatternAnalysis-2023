import torch, dataset, utils, modules, time, pickle
import matplotlib.pyplot as plt
from torch import optim
from tqdm import tqdm

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
    gen.train()
    critic.train()
    mapping_network.train() # Set model to train
    loop = tqdm(loader, leave=True)
    for batch_idx, real in enumerate(loop):
        real = real.to(utils.DEVICE)
        cur_batch_size = real.shape[0]

        w     = utils.get_w(mapping_network, cur_batch_size)
        noise = utils.get_noise(cur_batch_size)
        with torch.cuda.amp.autocast():
            fake = gen(w, noise)
            critic_fake = critic(fake.detach())
            
            critic_real = critic(real)
            gp = utils.gradient_penalty(critic, real, fake, device=utils.DEVICE)
            loss_critic = (
                -(torch.mean(critic_real) - torch.mean(critic_fake))
                + utils.LAMBDA_GP * gp
                + (0.001 * torch.mean(critic_real ** 2))
            )

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

        if batch_idx == 0:
            loss = loss_critic.item()
            #store first loss value for logging

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
    gen.eval()
    critic.eval()
    mapping_network.eval()
    total_loss = 0
    samples_validated = 0

    for real in loader:
        real = real.to(utils.DEVICE)
        cur_batch_size = real.shape[0]

        w     = utils.get_w(mapping_network, cur_batch_size)
        noise = utils.get_noise(cur_batch_size)
        with torch.cuda.amp.autocast():
            fake = gen(w, noise)
            critic_fake = critic(fake.detach())
            
            critic_real = critic(real)
            gp = utils.gradient_penalty(critic, real, fake, device=utils.DEVICE)
            loss_critic = (
                -(torch.mean(critic_real) - torch.mean(critic_fake))
                + utils.LAMBDA_GP * gp
                + (0.001 * torch.mean(critic_real ** 2))
            )

        total_loss += loss_critic.item()
        samples_validated += 1
    return total_loss / samples_validated

loader_train        = dataset.create_data_loader("train")
loader_validate        = dataset.create_data_loader("validate")

gen                 = modules.Generator(utils.LOG_RESOLUTION, utils.W_DIM).to(utils.DEVICE)
critic              = modules.Discriminator(utils.LOG_RESOLUTION).to(utils.DEVICE)
mapping_network     = modules.MappingNetwork(utils.Z_DIM, utils.W_DIM).to(utils.DEVICE)
path_length_penalty = modules.PathLengthPenalty(0.99).to(utils.DEVICE)

opt_gen             = optim.Adam(gen.parameters(), lr=utils.LEARNING_RATE, betas=(0.0, 0.99))
opt_critic          = optim.Adam(critic.parameters(), lr=utils.LEARNING_RATE, betas=(0.0, 0.99))
opt_mapping_network = optim.Adam(mapping_network.parameters(), lr=utils.LEARNING_RATE, betas=(0.0, 0.99))

losses = []
best_loss = float('inf')
best_model_state_dict = None
validate_every_n_epochs = 5
start_time = time.time()
last_epoch = start_time

load_checkpoint = False

starting_epoch = 1
if load_checkpoint:
    start_time = 1697569326.1386592
    with open(f'checkpoint_{start_time}.pickle', 'rb') as handle:
        starting_epoch, losses, best_loss = pickle.load(handle)
    gen.load_state_dict(torch.load(f"best_gen_{start_time}.pth"))
    critic.load_state_dict(torch.load(f'best_critic_{start_time}.pth'))
    mapping_network.load_state_dict(torch.load(f'best_map_{start_time}.pth'))
    #start_time += 0.0000000001
    starting_epoch+=1

for epoch in tqdm(range(starting_epoch, utils.epochs + 1)):

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
    # Save losses obtained from training the model
    losses.append(loss)
    print(f"Epoch {epoch:03d} | Training Loss: {loss}")
    if epoch % validate_every_n_epochs == 0:
        average_loss = eval_fn(
            critic,
            gen,
            mapping_network,
            loader_validate,
        )

        utils.generate_examples(mapping_network, gen, epoch, start_time)

        # Save model if better
        if average_loss < best_loss:
            best_loss = average_loss
            best_gen_state_dict = gen.state_dict()
            best_critic_state_dict = critic.state_dict()
            best_map_state_dict = mapping_network.state_dict()

            torch.save(best_gen_state_dict, f'best_gen_{start_time}.pth')
            torch.save(best_critic_state_dict, f'best_critic_{start_time}.pth')
            torch.save(best_map_state_dict, f'best_map_{start_time}.pth')
            with open(f'checkpoint_{start_time}.pickle', 'wb') as handle:
                pickle.dump([epoch, losses, best_loss], handle, protocol=pickle.HIGHEST_PROTOCOL)

            # Print validation results
            print(f"Epoch {epoch} | Validation Loss: {average_loss}")
    time_elapsed = time.time() - last_epoch
    print(f'Epoch {epoch} complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    #print(f"Time Running: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    last_epoch = time.time()

# Plot the losses over number of epochs
plt.plot(range(1, len(losses) + 1), losses, label='Training Loss')
plt.title('Training Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss_plot.png')
plt.close()