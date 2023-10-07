import torch
import torch.nn as nn
import timm
from torch.utils.tensorboard import SummaryWriter
import time
import os

import dataset as ds

# Initialise device
print("PyTorch Version:", torch.__version__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if (not torch.cuda.is_available()):
	print("Warning: Cuda not available, using CPU instead.")
 
# Hyper-parameters
numEpochs = 2500
learningRate = 0.01
gamma = 0.9
trainFromLastRun = True
savePath = "models/mnistClassifier.pth"

# Initialise Model
if trainFromLastRun and os.path.exists(savePath):
  model = torch.load(savePath)
else: # vit_base_patch32_384.augreg_in21k_ft_in1k
	model = timm.create_model("vit_small_patch16_224.augreg_in21k_ft_in1k", img_size=28, num_classes=len(ds.classes), in_chans=ds.channels)
model = model.to(device)

# Initialise logging to display tracking information in TensorBoard

# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter('runs/classifier_experiment_0')
addedGraph = False

# model info
print("Model No. of Parameters:", sum([param.nelement() for param in model.parameters()]))
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)

# Training the model
model.train()
print("> Training")
start = time.time()

totalStep = len(ds.trainloader)

try:
	for epoch in range(numEpochs):
		for i, (images, labels) in enumerate(ds.trainloader): # load a batch
			images = images.to(device)
			labels = labels.to(device)
		
			# Add a graph representation of the network to our TensorBoard
			if not addedGraph:
				writer.add_graph(model, images)
				writer.close()
				addedGraph = True

			# Forward Pass
			outputs = model(images)
			loss = criterion(outputs, labels)
			
			# Backward and optimize
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
		
			if ((i + 1) % 100 == 0):
				print("Epoch [{}/{}], Step[{}/{}], Loss: {:.5f}"
							.format(epoch+1, numEpochs, i+1, totalStep, loss.item()))
		
		# reduce the learning rate each epoch.
		scheduler.step()
except KeyboardInterrupt:
  pass
finally:
	end = time.time()
	elapsed = end - start
	print("Training took " + str(elapsed) + " secs")

	os.makedirs(os.path.dirname(savePath), exist_ok=True)
	torch.save(model, savePath)