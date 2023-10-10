import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from dataset import classes

"""
Various helper functions to display results from the training and testing
process, used to make TensorBoard displays.
(credit to https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html)
"""

# helper function to show an image
def matplotlib_imshow(img: torch.Tensor, one_channel=False):
	if one_channel:
		img = img.mean(dim=0)
	img = img / 2 + 0.5     # unnormalize
	npimg = img.detach().cpu().numpy()
	if one_channel:
		plt.imshow(npimg, cmap="Greys")
	else:
		plt.imshow(np.transpose(npimg, (1, 2, 0)))
  
# helper functions

def images_to_probs(model, images):
	'''
	Generates predictions and corresponding probabilities from a trained
	network and a list of images
	'''
	output = model(images)
	# convert output probabilities to predicted class
	_, preds_tensor = torch.max(output, 1)
	preds = np.squeeze(preds_tensor.detach().cpu().numpy())
	return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]


def plot_classes_preds(model, images, labels):
	'''
	Generates matplotlib Figure using a trained network, along with images
	and labels from a batch, that shows the network's top prediction along
	with its probability, alongside the actual label, coloring this
	information based on whether the prediction was correct or not.
	Uses the "images_to_probs" function.
	'''
	preds, probs = images_to_probs(model, images)
	# plot the images in the batch, along with predicted and true labels
	fig = plt.figure(figsize=(12, 48))
	for idx in np.arange(4):
		ax = fig.add_subplot(1, 4, idx+1, xticks=[], yticks=[])
		matplotlib_imshow(images[idx], one_channel=True)
		ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
				classes[preds[idx]],
				probs[idx] * 100.0,
				classes[labels[idx]]
    	),
			color=("green" if preds[idx]==labels[idx].item() else "red"))
	return fig