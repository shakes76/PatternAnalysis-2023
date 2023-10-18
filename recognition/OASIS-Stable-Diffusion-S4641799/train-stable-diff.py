import dataset, modules, utils, torch, time, pickle
import matplotlib.pyplot as plt
from torch.optim import Adam
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = modules.UNet()
model = model.to(device)

optimizer = Adam(model.parameters(), lr=0.001)

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
    model.load_state_dict(torch.load(f"best_model_{start_time}.pth"))
    #start_time += 0.0000000001
    starting_epoch+=1

for epoch in tqdm(range(starting_epoch, utils.epochs + 1)):

    model.train() # Set model to train
    #for step, batch in enumerate(dataset.create_data_loader("train")):
    for step, batch in enumerate(tqdm(dataset.create_data_loader("train"))):
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
            utils.sample_save_image(model, epoch, utils.output_dir, device, start_time)

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

            torch.save(best_model_state_dict, f'best_model_{start_time}.pth')
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