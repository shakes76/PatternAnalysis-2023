import time
import torch, torchvision

from dataset import Dataset, machine
from modules import Model_Generator, Model_Discriminator


def main():

    """ pytorch exe device """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"pytorch version: {torch.__version__}, exe device: {device}", flush=True)

    """ set the initial learning rates """
    generator_lr = 3e-5
    discriminator_lr = 2e-5

    """ run for 300 epochs on rangpur """
    epochs = 300 if machine == "rangpur" else 3

    """ load training dataset """
    train_loader = Dataset(train=True).loader()

    """ create the models and send to device """
    generator = Model_Generator().to(device)
    discriminator = Model_Discriminator().to(device)
    print(f"generator params: {sum([p.nelement() for p in generator.parameters()])}", flush=True)
    print(f"discriminator params: {sum([p.nelement() for p in discriminator.parameters()])}", flush=True)

    """ loss function: BCE loss for the discriminator """
    loss_function = torch.nn.BCELoss()

    """ optimiser and variable learning rate scheduller for generator """
    optimizer_generator = torch.optim.Adam(
        params=generator.parameters(), lr=generator_lr, betas=(0.5, 0.999),
    )
    lr_scheduler_generator = torch.optim.lr_scheduler.LambdaLR(
        optimizer_generator, lr_lambda=lambda epoch: 0.95**epoch,
    )

    """ optimiser and variable learning rate scheduller for discriminator """
    optimizer_discriminator = torch.optim.Adam(
        params=discriminator.parameters(), lr=discriminator_lr, betas=(0.5, 0.999),
    )
    lr_scheduler_discriminator = torch.optim.lr_scheduler.LambdaLR(
        optimizer_discriminator, lr_lambda=lambda epoch: 0.96**epoch,
    )

    """ start training loop """
    generator.train()
    discriminator.train()

    start = time.time()
    print("training...", flush=True)

    for epoch in range(epochs):
        for i, (images, labels) in enumerate(train_loader):

            """ send images and labels to hardware """
            images = images.to(device)
            labels = labels.to(device)

            """ downsample dataset and attempt to reconstruct using generator """
            downsampled = torchvision.transforms.Resize(60, antialias=True)(images).to(device)
            upsampled = generator(downsampled)
            
            """ train discriminator """
            discriminator_original = discriminator(images)
            discriminator_original_loss = loss_function(
                discriminator_original, torch.ones_like(discriminator_original),
            )
            discriminator_upsampled = discriminator(upsampled)
            discriminator_upsampled_loss = loss_function(
                discriminator_upsampled, torch.zeros_like(discriminator_upsampled),
            )
            discriminator_loss = (discriminator_original_loss + discriminator_upsampled_loss) / 2
            
            """ discriminator backward pass """
            discriminator.zero_grad()
            discriminator_loss.backward(retain_graph=True)
            optimizer_discriminator.step()

            """ train generator """
            outputs = discriminator(upsampled)
            generator_loss = loss_function(outputs, torch.ones_like(outputs))

            """ generator backwards pass """
            optimizer_generator.zero_grad()
            generator_loss.backward()
            optimizer_generator.step()

            """ print loss results """
            if (i + 1) % 300 == 0 or ((i + 1) % 10 == 0 and machine == "local"):
                print (
                    "epoch: %d/%d, step: %d/%d, d loss: %.5f, g loss: %.5f" % (
                        epoch + 1, epochs, i + 1, len(train_loader), 
                        discriminator_loss.item(), generator_loss.item(),
                    ), flush=True,
                )

        """ step the lr schedulers """
        lr_scheduler_generator.step()
        lr_scheduler_discriminator.step()
        print(
            "d lr: %.12f, g lr: %.12f" % (
                optimizer_discriminator.state_dict()["param_groups"][0]["lr"],
                optimizer_generator.state_dict()["param_groups"][0]["lr"],
            ), flush=True,
        )

        """ save the model """
        if (epoch + 1) % 30 == 0 or (machine == "local" and epoch + 1 == epochs):
            with open(
                file="models/sr_model_%03d.pt" % (epoch + 1), mode="wb") as f:
                torch.save(obj=generator.state_dict(), f=f)

    """ print training time """
    print(f"training time: {round((time.time() - start) / 60)} mins", flush=True)


if __name__ == "__main__":
    main()
