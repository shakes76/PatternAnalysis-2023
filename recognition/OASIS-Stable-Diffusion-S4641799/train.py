import dataset, modules, utils, torch, time, pickle
import matplotlib.pyplot as plt
from torch.optim import Adam

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create a model instance from module.py
model = modules.UNet()
model = model.to(device)

# Adam Optimizer for training the model
optimizer = Adam(model.parameters(), lr=0.001)

losses = [] # Initialize list for losses for plotting
best_loss = float('inf')  # Initialize with a high value
best_model_state_dict = None  # Variable to store the state_dict of the best model
validate_every_n_epochs = 5 # Variable to validate state of model every 5 epochs
since = time.time()
last_epoch = since

for epoch in range(utils.epochs):

    # Training Loop
    model.train()
    for step, batch in enumerate(dataset.create_data_loader("train")):
        optimizer.zero_grad()

        t = torch.randint(0, utils.T, (utils.BATCH_SIZE,), device=device).long()
        loss = utils.get_loss(model, batch, t, device)
        loss.backward()
        optimizer.step() 

        # Save losses obtained from training the model
        if step == 0:
            losses.append(loss.item())

        if epoch % 10 == 0 and step == 0:
            print(f"Epoch {epoch:03d} | step {step:03d} Loss: {loss.item()}")
            utils.sample_save_image(model, epoch, utils.output_dir, device)

    if epoch % validate_every_n_epochs == 0:
        model.eval()  # Set the model to evaluation mode
        total_loss = 0
        samples_validated = 0

        # Calculate average loss for current model
        with torch.no_grad():
            for step, batch in enumerate(dataset.create_data_loader("validate")):
                t = torch.randint(0, utils.T, (utils.BATCH_SIZE,), device=device).long()
                validation_loss = utils.get_loss(model, batch, t, device)
                total_loss += validation_loss.item() * utils.BATCH_SIZE
                samples_validated += utils.BATCH_SIZE

        
        average_loss = total_loss / samples_validated

        # Save model if better
        if average_loss < best_loss:
            best_loss = average_loss
            best_model_state_dict = model.state_dict()

            torch.save(best_model_state_dict, 'best_model.pth')
            with open('checkpoint.pickle', 'wb') as handle:
                pickle.dump([epoch, losses, best_loss], handle, protocol=pickle.HIGHEST_PROTOCOL)

        # Print validation results
        print(f"Epoch {epoch} | Validation Loss: {average_loss}")
    time_elapsed = time.time() - last_epoch
    print(f'Epoch {epoch} complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f"Time Running: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    last_epoch = time.time()

# Plot the losses over number of epochs
plt.plot(range(1, len(losses) + 1), losses, label='Training Loss')
plt.title('Training Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss_plot.png')
plt.close()