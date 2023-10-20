import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, CenterCrop, ToPILImage, Normalize, Resize
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from glob import glob


standardTransform = Compose([ToTensor(), CenterCrop(224), Normalize(0.11553987, 0.22542113)])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class ADNIDataset(Dataset):  
	"""Torch Dataset Wrapper for ADNI Dataset"""
	def __init__(self, img_data, transform=ToTensor()):
		self.transform = transform
		self.imgs = [self.transform(imgset[0]) for imgset in img_data]
		self.labels = [imgset[1] for imgset in img_data]
		
	def __len__(self):
		return len(self.imgs)
	
	def __getitem__(self, idx):
		return self.imgs[idx], self.labels[idx]

def load_images_from_directories(dirs, datasplit=0., verbose=False):
	"""Load images from data directory

	Args:
		dirs ([[string, int]]): List of directories and their labels
		datasplit (_type_, optional): The val to train split of the data. Defaults to 0..
		verbose (bool, optional): Verbosity of the function. Defaults to False.

	Returns:
		([string, int], [string, int]): A tuple of the train and val images and their labels
	"""
	image_paths = []
	train_paths = []
	val_paths = []
	train_images = []
	val_images = []
	# Iterate through each directory in list
	for dir in dirs:
		temp_image_paths = []
		# Get all image paths in directory
		for filename in glob(dir[0], recursive=True):
			temp_image_paths.append(filename)
			image_paths.append(filename)
		# Sort image paths by patient id
		temp_image_paths = sorted(temp_image_paths)
		curr_patient = ""
		count = 0
		# Load data and split into train and validation set
		for filename in temp_image_paths:
			patient_id = filename.split("/")[-1].split("_")[0]
			if (curr_patient != patient_id):
				curr_patient = patient_id
				count += 1
			if (datasplit != 0.):
				if (count % (round(1/datasplit)) == 0):
					count = 0
				if (count == 0):
					# Add to validation
					val_paths.append(filename)
					img = Image.open(filename)
					val_images.append([img.copy(), dir[1]])
					img.close()
				else:
					# Add to train
					train_paths.append(filename)
					img = Image.open(filename)
					train_images.append([img.copy(), dir[1]])
					img.close()
			else:
				train_paths.append(filename)
				img = Image.open(filename)
				train_images.append([img.copy(), dir[1]])
				img.close()
			
	if verbose:
		# Logging for debugging purposes
		print("Set1: ", train_paths[:2])
		print("Set2: ", val_paths[:2])
		for ip in image_paths[:10]:
			im = Image.open(ip[0])
			plt.figure()
			plt.imshow(im, cmap="gray")

	return train_images, val_images


def load_adni_images(datasplit = 0.2, verbose=False, local=True):
	"""Load adni images from data directory provided

	Args:
		datasplit (float, optional): The val to train split of the data. Defaults to 0.2.
		verbose (bool, optional): Verbosity of the function. Defaults to False.
		local (bool, optional): Whether training on PC or Rangpur. Defaults to True.

	Returns:
		_type_: _description_
	"""
	train_dirs = []
	test_dirs = []
 
	# If training on local machine
	if local:
		base_dir = "C:/Users/Jun Khai/Documents/Uni/Year 5 Sem 2/PatternAnalysis-2023/recognition/45816967_ADNI_Vision_Transformer/"

		train_dirs.append((f'{base_dir}data/ADNI_AD_NC_2D/AD_NC/train/AD/*', 1))
		train_dirs.append((f'{base_dir}data/ADNI_AD_NC_2D/AD_NC/train/NC/*', 0))
		test_dirs.append((f'{base_dir}data/ADNI_AD_NC_2D/AD_NC/test/AD/*', 1))
		test_dirs.append((f'{base_dir}data/ADNI_AD_NC_2D/AD_NC/test/NC/*', 0))
	
	# If training on Rangpur
	else:
		base_dir = "/home/groups/comp3710/ADNI/AD_NC/"

		train_dirs.append((f'{base_dir}train/AD/*', 1))
		train_dirs.append((f'{base_dir}train/NC/*', 0))
		test_dirs.append((f'{base_dir}test/AD/*', 1))
		test_dirs.append((f'{base_dir}test/NC/*', 0))
	
	return load_images_from_directories(train_dirs, datasplit=datasplit, verbose=verbose), load_images_from_directories(test_dirs, datasplit=0, verbose=verbose)

def generate_adni_datasets(datasplit = 0.2, verbose = False, local=True, test=False):
	"""_summary_

	Args:
		datasplit (float, optional): The val to train split of the data . Defaults to 0.2.
		verbose (bool, optional): Verbosity of the function. Defaults to False.
		local (bool, optional): Whether training on PC or Rangpur. Defaults to True.
		test (bool, optional): Set true if fake images are to be generated. Defaults to False.

	Returns:
		(torch.utils.data.Dataset, torch.utils.data.Dataset, torch.utils.data.Dataset)): _description_
	"""
 	# Load images
	train_imgs, test_imgs = load_adni_images(datasplit=datasplit, verbose = verbose, local=local)
	
 	# Generate fake images
	if test:
		tensor2pil = ToPILImage()
		train_imgs = [[[tensor2pil(torch.randn(1, 224, 224)), 0], [tensor2pil(torch.randn(1, 224, 224)), 1]], [[torch.randn(1, 224, 224), 0], [torch.randn(1, 224, 224), 1]]]
		test_imgs = [[[tensor2pil(torch.randn(1, 224, 224)), 0], [tensor2pil(torch.randn(1, 224, 224)), 1]], [[torch.randn(1, 224, 224), 0], [torch.randn(1, 224, 224), 1]]]
	
	# Create datasets
	train_set = ADNIDataset(train_imgs[0], transform=standardTransform)
	val_set = ADNIDataset(train_imgs[1], transform=standardTransform)
	test_set = ADNIDataset(test_imgs[0], transform=standardTransform)
	return train_set, val_set, test_set

def get_normalise_constants():
	""" Get mean and standard deviation of the train set

	Returns:
		(double, double): mean and standard deviation of the train set
	"""
	# Load images
	train_imgs, test_imgs, = load_adni_images(datasplit=0)
	transform = ToTensor()
	
	# Convert images from PIL to numpy array
	imgs = np.array([transform(imgset[0]).squeeze(0).numpy() for imgset in train_imgs[0]])
	print(imgs.mean(), imgs.std())
	
	# Calculate mean and standard deviation
	return imgs.mean(), imgs.std()