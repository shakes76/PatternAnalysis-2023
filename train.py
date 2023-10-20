from dataset import *
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy


def train_model():
    # Set processing to GPU
    if torch.cuda.is_available():
        print("Using GPU.")
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)
    
    # Loading in the data for the model
    dataset = torch.Tensor(load_brain_images())
    dataset = nn.functional.interpolate(dataset, (1, 64, 64))
    # dataset /= 4149

    for img in dataset:
        img /= img.max()
        img *= 2.0
        img -= 1.0


    # Making generator
    netG = StyleGANGenerator(z_dim, init_channels, init_resolution, device).to(device)

    # Making discriminator
    netD = Discriminator(1).to(device)

    # Loss function
    criterion = nn.BCELoss()

    # Optimizers
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.99))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.99))

    # Test output
    # netG.eval()
    # output = netG(torch.randn(1, 1, 1, 128))
    # plt.imshow(output.detach().numpy()[0][0])
    # plt.show()
    # netG.train()
    # netD.eval()
    # output = netD(output)
    # netD.train()

    # Length of dataset
    load_len = len(dataset)

    # Label for training discriminator
    real_target = torch.Tensor(batch_size, 1, 1, 1)
    real_target.apply_(lambda x: 1.0)
    fake_target = torch.Tensor(batch_size, 1, 1, 1)
    fake_target.apply_(lambda x: 0.0)

    for epoch in range(1, epochs + 1):
        print("Training Epoch:", epoch, "/", epochs)
        i = 0
        # Training models
        for data in dataset:
            # Training discriminator on real
            optimizerD.zero_grad()
            output = netD(data.to(device))
            loss = criterion(output, real_target.to(device))
            loss.backward()
            optimizerD.step()

            # Training discriminator on fake
            # Making fake image
            noise = torch.randn(batch_size, 1, 1, z_dim, device=device)
            fake_data = netG(noise)
            # Discriminating
            output = netD(fake_data.detach())
            err_fake = criterion(output, fake_target.to(device))
            err_fake.backward()
            optimizerD.step()

            # Training generator
            netG.zero_grad()
            output = netD(fake_data)
            err_gen = criterion(output, real_target.to(device))
            err_gen.backward()
            optimizerG.step()

            i += 1
            if i % 900 == 0:
                print("Progress: (", i, "/", int(load_len/batch_size), ")")
                # Test output for debugging purposes
                # netG.eval()
                # noise = torch.randn(batch_size, 1, 1, z_dim, device=device)
                # output = netG(noise)
                # plt.imshow(output.cpu().detach().numpy()[0][0])
                # plt.show()
                # netG.train()

        # Saving models after completion of epoch
        torch.save(netG.state_dict(), 'model_gen.pt')
        torch.save(netD.state_dict(), 'model_disc.pt')

    # Final outputs
    for _ in range(10):
        netG.eval()
        noise = torch.randn(batch_size, 1, 1, z_dim, device=device)
        output = netG(noise)
        plt.imshow(output.cpu().detach().numpy()[0][0])
        plt.show()

    return
