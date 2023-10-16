import torch
import time
import logging

import dataset as ds

# Setup logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

fh = logging.FileHandler('output/vit_out.txt')
fh.setLevel(logging.DEBUG) # or any level you want
logger.addHandler(fh)

# Initialise device
logger.debug(f"PyTorch Version: {torch.__version__}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if (not torch.cuda.is_available()):
	logger.warning("Warning: Cuda not available, using CPU instead.")

# Parameters
savePath = "models/vitClassifier.pth"

# Initialise Model
model = torch.load(savePath)
model = model.to(device)

# model info
logger.debug(f"Model No. of Parameters: {sum([param.nelement() for param in model.parameters()])}")
logger.debug(model)

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

end = time.time()
elapsed = end - start
logger.info("Testing took " + str(elapsed) + " secs")

logger.info("END")