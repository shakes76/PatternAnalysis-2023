import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, Resize, Normalize, PILToTensor, CenterCrop
import matplotlib.pyplot as plt
from PIL import Image
from glob import glob


# standardTransform = Compose([ToTensor(), Resize([128, 128], antialias=True), Normalize(0.5, 0.5, 0.5)])
standardTransform = Compose([PILToTensor(), CenterCrop(192)])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class ADNIDataset(Dataset):  
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
		dirs (_type_): _description_
	"""
	image_paths = []
	train_paths = []
	val_paths = []
	train_images = []
	val_images = []
	for dir in dirs:
		for filename in glob(dir[0], recursive=True):
			image_paths.append(filename)
	image_paths = sorted(image_paths)
	
	curr_patient = ""
	count = 0
	for ip in image_paths:
		patient_id = ip.split("/")[-1].split("_")[0]
		# print(patient_id)
		if (curr_patient != patient_id):
			curr_patient = patient_id
			count += 1
		if (datasplit != 0.):
			# print(patient_id)
			if (count % (round(1/datasplit)) == 0):
				count = 0
			if (count == 0):
				# print("val")
				val_paths.append(ip)
				img = Image.open(ip)
				val_images.append([img.copy(), dir[1]])
				img.close()
			else:
				# print("train")
				train_paths.append(ip)
				img = Image.open(ip)
				train_images.append([img.copy(), dir[1]])
				img.close()
		else:
			train_paths.append(ip)
			img = Image.open(ip)
			train_images.append([img.copy(), dir[1]])
			img.close()
			
	if verbose:
		print("Set1: ", train_paths[:2])
		print("Set2: ", val_paths[:2])
		for ip in image_paths[:10]:
			im = Image.open(ip)
			plt.figure()
			plt.imshow(im, cmap="gray")

	return train_images, val_images


def load_adni_images(datasplit = 0.2, verbose=False):
	"""Load oasis images from data directory provided

	Returns:
		[string, list[np.array]]: image paths and images
	"""
	train_dirs = []
	test_dirs = []

	base_dir = "C:/Users/Jun Khai/Documents/Uni/Year 5 Sem 2/PatternAnalysis-2023/recognition/45816967_ADNI_Vision_Transformer/"

	train_dirs.append((f'{base_dir}data/ADNI_AD_NC_2D/AD_NC/train/AD/*', 1))
	train_dirs.append((f'{base_dir}data/ADNI_AD_NC_2D/AD_NC/train/NC/*', 0))
	test_dirs.append((f'{base_dir}data/ADNI_AD_NC_2D/AD_NC/test/AD/*', 1))
	test_dirs.append((f'{base_dir}data/ADNI_AD_NC_2D/AD_NC/test/NC/*', 0))
		
	return load_images_from_directories(train_dirs, datasplit=datasplit, verbose=verbose), load_images_from_directories(test_dirs, datasplit=0, verbose=verbose)

