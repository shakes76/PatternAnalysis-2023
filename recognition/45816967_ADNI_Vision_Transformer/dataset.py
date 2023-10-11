from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
import matplotlib.pyplot as plt
from PIL import Image
from glob import glob


# standardTransform = Compose([ToTensor(), Resize([128, 128], antialias=True), Normalize(0.5, 0.5, 0.5)])
standardTransform = Compose([ToTensor(), Resize([128, 128], antialias=True)])

class ADNIDataset(Dataset):  
	def __init__(self, img_data, transform=ToTensor()):
		self.imgs = img_data[0:]
		self.labels = img_data[1:]
		self.transform = transform
		
	def __len__(self):
		return len(self.imgs)
	
	def __getitem__(self, idx):
		return self.transform(self.imgs[idx]), self.labels[idx]

def load_images_from_directories(dirs, verbose=False):
	"""Load images from data directory

	Args:
		dirs (_type_): _description_
	"""
	image_paths = []
	images = []
	for dir in dirs:
		for filename in glob(dir[0], recursive=True):
			image_paths.append(filename)
			img = Image.open(filename)
			images.append([img.copy(), dir[1]])
			img.close()
	image_paths = sorted(image_paths)

	if verbose:
		for ip in image_paths[:10]:
			im = Image.open(ip)
			plt.figure()
			plt.imshow(im, cmap="gray")

	return images


def load_adni_images(verbose=False):
	"""Load oasis images from data directory provided

	Returns:
		[string, list[np.array]]: image paths and images
	"""
	dataset_names = ["test", "test"]
	train_dirs = []
	test_dirs = []

	base_dir = "C:/Users/Jun Khai/Documents/Uni/Year 5 Sem 2/PatternAnalysis-2023/recognition/45816967_ADNI_Vision_Transformer/"

	train_dirs.append((f'{base_dir}data/ADNI_AD_NC_2D/AD_NC/train/AD/*', 1))
	train_dirs.append((f'{base_dir}data/ADNI_AD_NC_2D/AD_NC/train/NC/*', 0))
	test_dirs.append((f'{base_dir}data/ADNI_AD_NC_2D/AD_NC/test/AD/*', 1))
	test_dirs.append((f'{base_dir}data/ADNI_AD_NC_2D/AD_NC/test/NC/*', 0))
		
	return load_images_from_directories(train_dirs, verbose), load_images_from_directories(test_dirs, verbose)

  