from utils import *
from dataset import *
from modules import *

def train(trainloader, n_epochs, model, optimizer, criterion):
    """
    Train the model.
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
    model = train(trainloader, NUM_EPOCHS, model, optimizer, criterion)
    print("Training Time: %.2f min" % ((time.time() - train_start_time) / 60))
    torch.save(model.state_dict(), MODEL_PATH)

    print("Execution Time: %.2f min" % ((time.time() - start_time) / 60))

if __name__ == "__main__":
    main()

