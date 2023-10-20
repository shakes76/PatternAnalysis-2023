import torch
import torch.nn as nn
import timm
from torch.utils.tensorboard import SummaryWriter
import time
import os
import gc

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
logger.debug(f"PyTorch Version: {torch.__version__}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if (not torch.cuda.is_available()):
	logger.warning("Warning: Cuda not available, using CPU instead.")
 
# Hyper-parameters
numEpochs = 400
# Train at least this many epochs before
# letting the validation accuracy control
# whether to cutoff.
minEpochBeforeValidationCutoff = 10
validationEpochsToAverageOver = 4
# If we've been training for more than 8 hrs we
# should stop so we don't get force-quit by the scheduler.
maxTrainTime = 8 * 60 * 60
learningRate = 0.01
gamma = 0.99
trainFromLastRun = False
savePath = "models/vitClassifier.pth"
numBatchesBetweenLogging = 100

minEpochBeforeValidationCutoff = max(minEpochBeforeValidationCutoff, validationEpochsToAverageOver * 2)

# Initialise Model
if trainFromLastRun and os.path.exists(savePath):
  model = torch.load(savePath)
else:
  # Apparently better to choose a pre-trained model that is lower resolution than the
	# fine-tune database's images, i.e. for this choose a 224 model.
	# Probably vit_base_patch32_224.augreg_in21k_ft_in1k
	# trying vit_tiny_patch16_224.augreg_in21k_ft_in1k for less parameters
	# 256 x 240
	model = timm.create_model("vit_base_patch32_224.augreg_in21k_ft_in1k", img_size=256, num_classes=len(ds.classes), in_chans=ds.channels)
model = model.to(device)

for param in model.parameters():
	param.requires_grad = False

for param in model.head.parameters():
	param.requires_grad = True

# TODO: Look into the symposium model, might be better for this application

# Initialise logging to display tracking information in TensorBoard

# default `log_dir` is "runs" - we'll be more specific here
id = 0
for directory in os.listdir('runs'):
	if not os.path.isdir(directory):
		continue
	if int(directory.split("_")[-1]) >= id:
		id = int(directory.split("_")[-1]) + 1

writer = SummaryWriter(f"runs/classifier_experiment_{id}")
addedGraph = False

# model info
logger.debug(f"Model No. of Parameters: {sum([param.nelement() for param in model.parameters()])}")
logger.debug(model)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)

totalStep = len(ds.trainloader)

# Training the model
model.train()
logger.info(f"> Training, at most {numEpochs} epochs with {totalStep} steps per epoch.")
start = time.time()

running_loss = 0.0
validation_accuracies = []

for epoch in range(numEpochs):	
	for i, (images, labels) in enumerate(ds.trainloader): # load a batch
		images: torch.Tensor = images.to(device)
		labels: torch.Tensor = labels.to(device)
	
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
										epoch * totalStep + i)

			# log a Matplotlib Figure showing the model's predictions on a
			# random mini-batch
			writer.add_figure('predictions vs. actuals',
											plotting.plot_classes_preds(model, images, labels),
											global_step=epoch * totalStep + i)

			logger.info("Epoch [{}/{}], Step[{}/{}], Average Loss: {:.5f}"
						.format(epoch+1, numEpochs, i+1, totalStep, running_loss / numBatchesBetweenLogging))

			running_loss = 0.0

		del images, labels
		gc.collect()
		torch.cuda.empty_cache()

		# if more than maxTrainTime seconds have passed since training started, exit out
		# of all loops to prevent the training progress from being lost.
		if time.time() - start > maxTrainTime:
			break
	
	validation_total = 0
	validation_correct = 0
	for i, (images, labels) in enumerate(ds.validloader):
		images = images.to(device)
		labels = labels.to(device)
  
		outputs = model(images)
		_, predicted = torch.max(outputs.data, 1)
	
		validation_total += labels.size(0)
		validation_correct += (predicted == labels).sum().item()

		del images, labels
		gc.collect()
		torch.cuda.empty_cache()
	
	if validation_total > 0:
		# Update validation accuracy list, do cutoff based on it, and add that info to the writer
		validation_accuracies.append(validation_correct / validation_total)

		writer.add_scalar("validation accuracy", validation_accuracies[-1], global_step=epoch * totalStep + i)
		logger.info("Epoch [{}/{}], Validation Accuracy: {:.5f}"
					.format(epoch+1, numEpochs, validation_accuracies[-1]))

		# If at least minEpochBeforeValidationCutoff epochs have been finished, and the average validation accuracy in
		# the last validationEpochsToAverageOver epochs was less than the average accuracy of the previous
		# validationEpochsToAverageOver, exit the training since the model is starting to overfit.
		if epoch + 1 > minEpochBeforeValidationCutoff and \
				sum(validation_accuracies[-2*validationEpochsToAverageOver:-validationEpochsToAverageOver]) > \
					sum(validation_accuracies[:-validationEpochsToAverageOver]):
			logger.info("Exiting early since accuracy on validation set has started dropping.")
			break

	if time.time() - start > maxTrainTime:
		logger.info("Exiting early to prevent from running overtime.")
		break
	
	should_stop = False
	with open("should_stop.txt", "r") as file:
		if file.readline().strip().upper() == "STOP":
			logger.info("Exiting early on user request.")
			should_stop = True
	if should_stop:
		with open("should_stop.txt", "w") as file:
			file.write("")
		break
	
	# reduce the learning rate each epoch.
	scheduler.step()

end = time.time()
elapsed = end - start
logger.info("Training took " + str(elapsed) + " secs")

# Test the model
logger.info("> Testing")
start =  time.time()
model.eval()


with torch.no_grad():
	correct = 0
	total = 0
	
	for images, labels in ds.trainloader:
		images = images.to(device)
		labels = labels.to(device)
  
		outputs = model(images)
		_, predicted = torch.max(outputs.data, 1)
	
		total += labels.size(0)
		correct += (predicted == labels).sum().item()
	logger.info("Test Accuracy: {:.5f} %".format(100 * correct / total))
	writer.add_scalar("test accuracy", correct / total)

end = time.time()
elapsed = end - start
logger.info("Testing took " + str(elapsed) + " secs")

os.makedirs(os.path.dirname(savePath), exist_ok=True)
torch.save(model, savePath)

writer.close()

logger.info("END")