import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, PILToTensor, CenterCrop, ToPILImage
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
	print(dirs)
	for dir in dirs:
		temp_image_paths = []
		for filename in glob(dir[0], recursive=True):
			temp_image_paths.append(filename)
			image_paths.append(filename)
		temp_image_paths = sorted(temp_image_paths)
		curr_patient = ""
		count = 0
		for filename in temp_image_paths:
			patient_id = filename.split("/")[-1].split("_")[0]
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
					val_paths.append(filename)
					img = Image.open(filename)
					val_images.append([img.copy(), dir[1]])
					img.close()
				else:
					# print("train")
					train_paths.append(filename)
					img = Image.open(filename)
					train_images.append([img.copy(), dir[1]])
					img.close()
			else:
				train_paths.append(filename)
				img = Image.open(filename)
				train_images.append([img.copy(), dir[1]])
				img.close()
	# image_paths = sorted(image_paths)
	
	# for ip in image_paths:
		
			
	if verbose:
		print("Set1: ", train_paths[:2])
		print("Set2: ", val_paths[:2])
		for ip in image_paths[:10]:
			im = Image.open(ip[0])
			plt.figure()
			plt.imshow(im, cmap="gray")

	return train_images, val_images


def load_adni_images(datasplit = 0.2, verbose=False, local=True):
	"""Load oasis images from data directory provided

	Returns:
		[string, list[np.array]]: image paths and images
	"""
	train_dirs = []
	test_dirs = []

	if local:
		base_dir = "C:/Users/Jun Khai/Documents/Uni/Year 5 Sem 2/PatternAnalysis-2023/recognition/45816967_ADNI_Vision_Transformer/"

		train_dirs.append((f'{base_dir}data/ADNI_AD_NC_2D/AD_NC/train/AD/*', 1))
		train_dirs.append((f'{base_dir}data/ADNI_AD_NC_2D/AD_NC/train/NC/*', 0))
		test_dirs.append((f'{base_dir}data/ADNI_AD_NC_2D/AD_NC/test/AD/*', 1))
		test_dirs.append((f'{base_dir}data/ADNI_AD_NC_2D/AD_NC/test/NC/*', 0))
	
	else:
		base_dir = "/home/groups/comp3710/ADNI/AD_NC/"

		train_dirs.append((f'{base_dir}train/AD/*', 1))
		train_dirs.append((f'{base_dir}train/NC/*', 0))
		test_dirs.append((f'{base_dir}test/AD/*', 1))
		test_dirs.append((f'{base_dir}test/NC/*', 0))
  
	print(train_dirs)
	print(test_dirs)
	
	return load_images_from_directories(train_dirs, datasplit=datasplit, verbose=verbose), load_images_from_directories(test_dirs, datasplit=0, verbose=verbose)

def generate_adni_datasets(datasplit = 0.2, verbose = False, local=True, test=False):
	train_imgs, test_imgs = load_adni_images(datasplit=datasplit, verbose = verbose, local=local)
	
	if test:
		tensor2pil = ToPILImage()
		train_imgs = [[[tensor2pil(torch.randn(1, 192, 192)), 0], [tensor2pil(torch.randn(1, 192, 192)), 1]], [[torch.randn(1, 192, 192), 0], [torch.randn(1, 192, 192), 1]]]
		test_imgs = [[[tensor2pil(torch.randn(1, 192, 192)), 0], [tensor2pil(torch.randn(1, 192, 192)), 1]], [[torch.randn(1, 192, 192), 0], [torch.randn(1, 192, 192), 1]]]
	train_set = ADNIDataset(train_imgs[0], transform=standardTransform)
	val_set = ADNIDataset(train_imgs[1], transform=standardTransform)
	test_set = ADNIDataset(test_imgs[0], transform=standardTransform)
	return train_set, val_set, test_set