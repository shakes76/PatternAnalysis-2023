from utils import *
from dataset import *
from modules import *

def train_vae(trainloader, n_epochs, model, optimizer, criterion):
    """
    Train the VAE model.
    """
    # Start training
    for epoch in range(n_epochs):
        loop = tqdm(enumerate(trainloader))
        for i, (x, y) in loop:
            # Forward pass
            x = x.to(device).view(-1, INPUT_DIM) # -1: auto decide
            x_reconst, mu, sigma = model(x)

            # loss, formulas
            reconst_loss = criterion(x_reconst, x)                                          # Reconstruction Loss
            kl_div = - torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))    # Kullback-Leibler Divergence
            loss = reconst_loss + kl_div

            # Backprop and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss.item())
        print ('Epoch [{}/{}], Loss: {:.4f}' 
                    .format(epoch+1, n_epochs, loss.item()))

    return model

def train_vqvae(epoch, loader, model, optimizer, scheduler, device):
    if dist.is_primary():
        loader = tqdm(loader)

    criterion = nn.MSELoss()

    latent_loss_weight = 0.25
    sample_size = 25

    mse_sum = 0
    mse_n = 0

    for i, (img, label) in enumerate(loader):
        model.zero_grad()

        img = img.to(device)

        out, latent_loss = model(img)
        recon_loss = criterion(out, img)
        latent_loss = latent_loss.mean()
        loss = recon_loss + latent_loss_weight * latent_loss
        loss.backward()

        if scheduler is not None:
            scheduler.step()
        optimizer.step()

        part_mse_sum = recon_loss.item() * img.shape[0]
        part_mse_n = img.shape[0]
        comm = {"mse_sum": part_mse_sum, "mse_n": part_mse_n}
        comm = dist.all_gather(comm)

        for part in comm:
            mse_sum += part["mse_sum"]
            mse_n += part["mse_n"]

        if dist.is_primary():
            lr = optimizer.param_groups[0]["lr"]

            loader.set_description(
                (
                    f"epoch: {epoch + 1}; mse: {recon_loss.item():.5f}; "
                    f"latent: {latent_loss.item():.3f}; avg mse: {mse_sum / mse_n:.5f}; "
                    f"lr: {lr:.5f}"
                )
            )

            if i % 100 == 0:
                model.eval()

                sample = img[:sample_size]

                with torch.no_grad():
                    out, _ = model(sample)

                utils.save_image(
                    torch.cat([sample, out], 0),
                    f"sample/{str(epoch + 1).zfill(5)}_{str(i).zfill(5)}.png",
                    nrow=sample_size,
                    normalize=True,
                    range=(-1, 1),
                )

                model.train()



def main():
    """
    train & save the model
    """
    start_time = time.time()
    print("Program Starts")
    print("Device:", device)

    # Data
    print("Loading Data...")
    trainloader, validloader = load_data(test=False)
    
    # Model, Loss, Optmizer
    model = VAE(INPUT_DIM, Z_DIM, H_DIM).to(device)
    criterion = nn.BCELoss(reduction="sum") # loss_func
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_RATE)

    # Train
    train_start_time = time.time()
    model = train_vae(trainloader, NUM_EPOCHS, model, optimizer, criterion)
    print("Training Time: %.2f min" % ((time.time() - train_start_time) / 60))
    torch.save(model.state_dict(), MODEL_PATH)

    print("Execution Time: %.2f min" % ((time.time() - start_time) / 60))

if __name__ == "__main__":
    main()

