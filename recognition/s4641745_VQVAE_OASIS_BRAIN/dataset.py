import torch as t
import torch.utils.data
import torchvision as tv
import torchvision.transforms as transforms
import os
from PIL import Image

# PATH = '/home/groups/comp3710'
PATH = os.path.dirname(__file__)
IMAGE_PATH = os.path.join(PATH, 'assets/images')
MODEL_PATH = os.path.join(PATH, 'assets/models')
BATCH_SIZE = 32

transform = transforms.Compose([transforms.ToTensor()])

vqvae_trainset = tv.datasets.ImageFolder(os.path.join(IMAGE_PATH, 'keras_png_slices_data/keras_png_slices_train'),
                                         transform=transform)
vqvae_train_loader = t.utils.data.DataLoader(vqvae_trainset, batch_size=BATCH_SIZE, shuffle=True)
vqvae_testset = tv.datasets.ImageFolder(os.path.join(IMAGE_PATH, 'keras_png_slices_data/keras_png_slices_test'),
                                        transform=transform)
vqvae_test_loader = t.utils.data.DataLoader(vqvae_testset, batch_size=BATCH_SIZE, shuffle=True)


class GanDataset(t.utils.data.Dataset):
    def __init__(self, model, transforms):
        self.model = model
        self.img_folder = os.path.join(IMAGE_PATH, 'keras_png_slices_data/keras_png_slices_train/no_label')
        self.images = os.listdir(self.img_folder)
        self.tfs = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, x):
        img_path = os.path.join(self.img_folder, self.images[x])
        img = Image.open(img_path).convert('RGB')
        img = self.tfs(img).unsqueeze(dim=0).to('cuda')
        encoded_output = self.model.encoder(img)
        z = self.model.conv1(encoded_output)
        _, _, _, z = self.model.vq(z)
        z = z.float().to('cuda').view(64, 64)
        z = torch.stack((z, z, z), 0) # GAN uses 3 channel inputs
        return z, z
