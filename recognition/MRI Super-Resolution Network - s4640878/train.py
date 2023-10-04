import time
import torch, torchvision

from dataset import Dataset, machine
from modules import Model_Generator, Model_Discriminator


def main():

    model_type = "gan"

    """ pytorch exe device """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device, flush=True)

    """ training params """
    lr = 3e-5
    epochs = 300 if machine == "rangpur" else 3

    """ load training dataset """
    train_loader = Dataset(train=True).loader()

    """ models """
    generator = Model_Generator().to(device)
    discriminator = Model_Discriminator().to(device)
    print(f"generator params: {sum([p.nelement() for p in generator.parameters()])}", flush=True)
    print(f"discriminator params: {sum([p.nelement() for p in discriminator.parameters()])}", flush=True)

    """ loss function """
    loss_function = torch.nn.MSELoss(
    ) if model_type == "cnn" else torch.nn.BCELoss(
    ) if model_type == "gan" else None

    """ optimiser for generator """
    optimizer_generator = torch.optim.SGD(
        params=generator.parameters(), lr=lr, 
        momentum=0.9, weight_decay=5e-4,
    ) if model_type == "cnn" else torch.optim.Adam(
        params=generator.parameters(), lr=lr, betas=(0.5, 0.999),
    ) if model_type == "gan" else None

    """ optimiser for discriminator """
    optimizer_discriminator = torch.optim.SGD(
        params=discriminator.parameters(), lr=lr, 
        momentum=0.9, weight_decay=5e-4,
    ) if model_type == "cnn" else torch.optim.Adam(
        params=discriminator.parameters(), lr=lr, betas=(0.5, 0.999),
    ) if model_type == "gan" else None

    """ variable learning rate scheduller for generator """
    lr_scheduler_generator = torch.optim.lr_scheduler.OneCycleLR(
        optimizer_generator, max_lr=lr, 
        steps_per_epoch=len(train_loader), epochs=epochs,
    ) if model_type == "cnn" else torch.optim.lr_scheduler.LambdaLR(
        optimizer_generator, lr_lambda=lambda epoch: 0.95**epoch,
    ) if model_type == "gan" else None

    """ variable learning rate scheduller for discriminator """
    lr_scheduler_discriminator = torch.optim.lr_scheduler.OneCycleLR(
        optimizer_discriminator, max_lr=lr, 
        steps_per_epoch=len(train_loader), epochs=epochs,
    ) if model_type == "cnn" else torch.optim.lr_scheduler.LambdaLR(
        optimizer_discriminator, lr_lambda=lambda epoch: 0.95**epoch,
    ) if model_type == "gan" else None

    """ train the model """
    generator.train()
    discriminator.train()

    start = time.time()
    print("training...", flush=True)

    for epoch in range(epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            downsampled = torchvision.transforms.Resize(60, antialias=True)(images).to(device)
            upsampled = generator(downsampled)
            
            """ train discriminator """
            if model_type == "gan": 
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
            if model_type == "gan": outputs = discriminator(upsampled)
            if model_type == "cnn": outputs = upsampled
            generator_loss = loss_function(
                outputs, images
            ) if model_type == "cnn" else loss_function(
                outputs, torch.ones_like(outputs)
            ) if model_type == "gan" else None

            """ generator backwards pass """
            optimizer_generator.zero_grad()
            generator_loss.backward()
            optimizer_generator.step()

            """ print loss results """
            if (i + 1) % 300 == 0 or ((i + 1) % 10 == 0 and machine == "local"):
                if model_type == "gan": print (
                        "epoch: %d/%d, step: %d/%d, d loss: %.5f, g loss: %.5f" % (
                            epoch + 1, epochs, i + 1, len(train_loader), 
                            discriminator_loss.item(), generator_loss.item(),
                        ), flush=True,
                    )
                if model_type == "cnn": print(
                        "epoch: %d/%d, step: %d/%d, loss: %.5f" % (
                            epoch + 1, epochs, i + 1, len(train_loader), 
                            generator_loss.item(),
                        ), flush=True,
                    )
         
            """ step the lr scheduler for cnn """
            if model_type == "cnn":
                lr_scheduler_generator.step()

        """ step the lr schedulers for gan """
        if model_type == "gan":
            lr_scheduler_generator.step()
            lr_scheduler_discriminator.step()
            print(
                "d lr: %.9f, g lr: %.9f" % (
                    optimizer_discriminator.state_dict()["param_groups"][0]["lr"],
                    optimizer_generator.state_dict()["param_groups"][0]["lr"],
                ), flush=True,
            )

        """ save the model """
        if (epoch + 1) % 30 == 0 or (machine == "local" and epoch + 1 == epochs):
            with open(
                file="models/sr_model_%03d.pt" % (epoch + 1), mode="wb") as f:
                torch.save(obj=generator.state_dict(), f=f)

    print(f"training time: {round((time.time() - start) / 60)} mins", flush=True)


if __name__ == "__main__":
    main()
