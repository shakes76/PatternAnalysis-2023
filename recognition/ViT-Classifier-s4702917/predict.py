import torch
import time
import os
import sys

import dataset as ds

# Don't buffer prints
sys.stdout.reconfigure(line_buffering=True, write_through=True)

# Initialise device
print("PyTorch Version:", torch.__version__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if (not torch.cuda.is_available()):
	print("Warning: Cuda not available, using CPU instead.")

# Parameters
savePath = "models/mnistClassifier.pth"

# Initialise Model
model = torch.load(savePath)
model = model.to(device)

# model info
print("Model No. of Parameters:", sum([param.nelement() for param in model.parameters()]))
print(model)

# Test the model
print("> Testing")
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
	print("Test Accuracy: {:.5f} %".format(100 * correct / total))

end = time.time()
elapsed = end - start
print("Testing took " + str(elapsed) + " secs")

print("END")