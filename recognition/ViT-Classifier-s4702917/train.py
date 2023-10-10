import torch
import torch.nn as nn
import timm
from torch.utils.tensorboard import SummaryWriter
import time
import os

import logging
import plotting
import dataset as ds

# Setup logging
logger = logging.getLogger("MAIN_LOGGER")
logger.setLevel(logging.DEBUG)

fh = logging.FileHandler('output/vit_out.txt')
fh.setLevel(logging.DEBUG) # or any level you want
logger.addHandler(fh)

# Initialise device
logger.debug("PyTorch Version:", torch.__version__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if (not torch.cuda.is_available()):
	logger.warning("Warning: Cuda not available, using CPU instead.")
 
# Hyper-parameters
numEpochs = 40
# If we've been training for more than 1.5 hrs we
# should stop so we don't get force-quit by the scheduler.
maxTrainTime = 1.5 * 60 * 60
learningRate = 0.01
gamma = 0.9
trainFromLastRun = False
savePath = "models/mnistClassifier.pth"
numBatchesBetweenLogging = 100

# Initialise Model
if trainFromLastRun and os.path.exists(savePath):
  model = torch.load(savePath)
else:
  # Apparently better to choose a pre-trained model that is lower resolution than the
	# fine-tune database's images, i.e. for this choose a 224 model.
	# Probably vit_base_patch32_224.augreg_in21k_ft_in1k
	# 256 x 240
	model = timm.create_model("vit_base_patch32_224.augreg_in21k_ft_in1k", img_size=256, num_classes=len(ds.classes), in_chans=ds.channels)
model = model.to(device)

# Initialise logging to display tracking information in TensorBoard

# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter('runs/classifier_experiment_0')
addedGraph = False

# model info
logger.debug("Model No. of Parameters:", sum([param.nelement() for param in model.parameters()]))
logger.debug(model)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)

totalStep = len(ds.trainloader)

# Training the model
model.train()
logger.info(f"> Training, at most {numEpochs} with {totalStep} steps per epoch.")
start = time.time()

running_loss = 0.0

for epoch in range(numEpochs):
	for i, (images, labels) in enumerate(ds.trainloader): # load a batch
		images = images.to(device)
		labels = labels.to(device)
	
		# Add a graph representation of the network to our TensorBoard
		if not addedGraph:
			writer.add_graph(model, images)
			writer.flush()
			addedGraph = True

		# Forward Pass
		outputs = model(images)
		loss = criterion(outputs, labels)
		
		# Backward and optimize
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		running_loss += loss.item()
	
		if ((i + 1) % numBatchesBetweenLogging == 0):
			# log the average loss for the past i batches
			writer.add_scalar("training loss",
										running_loss / numBatchesBetweenLogging,
										epoch * len(ds.trainloader) + i)

			# log a Matplotlib Figure showing the model's predictions on a
			# random mini-batch
			writer.add_figure('predictions vs. actuals',
											plotting.plot_classes_preds(model, images, labels),
											global_step=epoch * len(ds.trainloader) + i)

			logger.info("Epoch [{}/{}], Step[{}/{}] Average Loss: {:.5f}"
						.format(epoch+1, numEpochs, i+1, totalStep, running_loss / numBatchesBetweenLogging))

			running_loss = 0.0

			# if more than maxTrainTime seconds have passed since training started, exit out
			# of all loops to prevent the training progress from being lost.
			if time.time() - start > maxTrainTime:
				break
		if time.time() - start > maxTrainTime:
				break
	
	# reduce the learning rate each epoch.
	scheduler.step()

end = time.time()
elapsed = end - start
logger.info("Training took " + str(elapsed) + " secs")

os.makedirs(os.path.dirname(savePath), exist_ok=True)
torch.save(model, savePath)

writer.close()