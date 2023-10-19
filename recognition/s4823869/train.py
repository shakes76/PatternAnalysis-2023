import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import get_ttv, normalise
from modules import build_vqvae, build_pcnn
from torch.autograd import Variable

# Constants
LATENT_DIMENSION_SIZE = 8
NUM_EMBEDDINGS = 64
BETA = 0.25
VQVAE_EPOCHS = 100
PCNN_EPOCHS = 100
BATCH_SIZE = 128
MODEL_PATH = "vqvae.pth"
PCNN_PATH = "pcnn.pth"

def train_vqvae(train_set, train_vnce):
    # Create VQVAE model
    vqvae = build_vqvae(LATENT_DIMENSION_SIZE, NUM_EMBEDDINGS, BETA)
    vqvae = vqvae.cuda()  # Move model to GPU
    optimizer = optim.Adam(vqvae.parameters())

    # Define loss function
    criterion = nn.MSELoss()

    # Training loop
    for epoch in range(VQVAE_EPOCHS):
        vqvae.train()
        total_loss = 0.0
        for data in train_set:
            inputs = Variable(data).cuda()
            optimizer.zero_grad()
            outputs = vqvae(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_set)}")

    # Save the trained model
    torch.save(vqvae.state_dict(), MODEL_PATH)
    print("VQVAE model saved")

def train_pcnn(train_set, vqvae):
    # Create PCNN model
    pcnn = build_pcnn(vqvae)
    pcnn = pcnn.cuda()  # Move model to GPU
    optimizer = optim.Adam(pcnn.parameters())

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(PCNN_EPOCHS):
        pcnn.train()
        total_loss = 0.0
        for data in train_set:
            inputs = Variable(data).cuda()
            optimizer.zero_grad()
            logits = pcnn(inputs)
            loss = criterion(logits, inputs)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_set)}")

    # Save the trained model
    torch.save(pcnn.state_dict(), PCNN_PATH)
    print("PCNN model saved")

def main():
    # Load and preprocess the data
    tr, _, _ = get_ttv()
    train_set = DataLoader(normalise(tr), batch_size=BATCH_SIZE, shuffle=True)

    # Train the VQVAE
    train_vqvae(train_set, tr.var())

    # Train the PCNN
    vqvae = build_vqvae(LATENT_DIMENSION_SIZE, NUM_EMBEDDINGS, BETA)
    vqvae.load_state_dict(torch.load(MODEL_PATH))
    train_pcnn(train_set, vqvae)

if __name__ == "__main__":
    main()
