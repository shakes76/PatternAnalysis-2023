from modules import *
from utils import *
from dataset import *
import torch
import time

# -------
# Initialise device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if not torch.cuda.is_available():
    print("Warning CUDA not Found. Using CPU")

model = SuperResolution().to(device)

# -------
# Generate dataloaders
train_loader = generate_train_loader()
test_loader = generate_test_loader()

# model info
print("Model No. of Parameters:", sum([param.nelement() for param in model.parameters()]))
print(model)

# -------
# Train the model
total_step = len(train_loader)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

model.train()
print("> Training")
start = time.time()

for epoch in range(num_epochs):
    for i, (images, _) in enumerate(train_loader):

        low_res_images = resize_tensor(images)
        low_res_images = low_res_images.to(device)
        images = images.to(device)
        # Forward pass
        outputs = model(low_res_images)
        loss = criterion(outputs, images)

        # Backward propogration and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 10 == 0:
            print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.5f}"
                    .format(epoch+1, num_epochs, i+1, total_step, loss.item()), flush=True)
            

end = time.time()
elapsed = end - start
print("Training took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins in total") 

torch.save(model, model_path)
