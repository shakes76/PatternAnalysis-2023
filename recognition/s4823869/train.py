import torch
import torch.optim as optim
from skimage.metrics import structural_similarity as ssim
import numpy as np
from numpy.random import choice
import random
from matplotlib import pyplot as plt

from dataset import get_ttv, normalise
from modules import VectorQuantizer, PixelConvolution, ResidualBlock, Trainer

LATENT_DIMENSION_SIZE = 8
NUM_EMBEDDINGS = 64
BETA = 0.25
VQVAE_EPOCHS = 100
PCNN_EPOCHS = 100
BATCH_SIZE = 128
PCNN_OPT = 3e-4
VALIDATION_SPLIT = 0.1
MODEL_PATH = "vqvae.pth"
PCNN_PATH = "pcnn.pth"


def train_vqvae(train_set, train_vnce):
    trainer = Trainer(train_vnce, LATENT_DIMENSION_SIZE, NUM_EMBEDDINGS, BETA)
    trainer.optimizer = optim.Adam(trainer.parameters())

    training_losses = []

    for epoch in range(VQVAE_EPOCHS):
        for batch in train_set:
            batch = torch.tensor(batch, dtype=torch.float)  # Convert batch to PyTorch tensor
            trainer.zero_grad()
            recons = trainer(batch)
            loss = trainer.loss(recons, batch)
            loss.backward()
            trainer.optimizer.step()
            training_losses.append(loss.item())

    plt.plot(training_losses)
    plt.title("VQVAE Training Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

    return trainer.vqvae, trainer



def pcnn_graph(first, second, metric):
    plt.plot(first)
    plt.plot(second)
    plt.title(f"PCNN {metric} per Epoch")
    plt.ylabel(metric)
    plt.xlabel("Epoch")
    plt.legend(["Training", "Validation"], loc="upper right")
    plt.show()


def pcnn_metrics(metrics):
    tloss, vloss = metrics["loss"], metrics["val_loss"]
    tacc, vacc = metrics["accuracy"], metrics["val_accuracy"]
    pcnn_graph(tloss, vloss, "Loss")
    pcnn_graph(tacc, vacc, "Accuracy")


def main():
    tr, te, va = get_ttv()
    vnce = np.var(tr)
    train = normalise(tr)
    test = normalise(te)

    vqvae, trainer = train_vqvae(train, vnce)
    torch.save(vqvae.state_dict(), MODEL_PATH)

    encoder = trainer.encoder
    encoded_out = encoder(torch.tensor(test))  # Convert NumPy array to a PyTorch tensor
    encoded_out = encoded_out[:len(encoded_out) // 2]  # Remove half of them due to memory constraints

    qtiser = trainer.quantiser
    flat_encs = encoded_out.view(encoded_out.shape[0], -1)
    codebooks = qtiser.code_indices(flat_encs)
    codebooks = codebooks.numpy().reshape(encoded_out.shape[:-1])

    pcnn = build_pcnn(trainer, encoded_out)
    pcnn_optimizer = optim.Adam(pcnn.parameters(), lr=PCNN_OPT)
    pcnn_criterion = torch.nn.CrossEntropyLoss()
    pcnn_metrics = []

    for epoch in range(PCNN_EPOCHS):
        pcnn.train()
        for i in range(0, len(codebooks), BATCH_SIZE):
            pcnn_optimizer.zero_grad()
            batch = codebooks[i:i + BATCH_SIZE]
            inputs = torch.from_numpy(batch).long()
            targets = torch.from_numpy(batch).long()
            outputs = pcnn(inputs)
            loss = pcnn_criterion(outputs, targets)
            loss.backward()
            pcnn_optimizer.step()
        pcnn_metrics.append(loss.item())

    pcnn_graph(pcnn_metrics, pcnn_metrics, "Loss")
    torch.save(pcnn.state_dict(), PCNN_PATH)


if __name__ == "__main__":
    main()
