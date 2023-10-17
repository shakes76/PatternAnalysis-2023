import torch as t
import torch.utils.data
import torchvision as tv
import torchvision.transforms as transforms

PATH = '/home/groups/comp3710'
BATCH_SIZE = 32

transform = transforms.Compose([transforms.ToTensor()])

vqvae_trainset = tv.datasets.ImageFolder(PATH + '/keras_png_slices_train', transform=transform)
vqvae_train_loader = t.utils.data.DataLoader(vqvae_trainset, batch_size=BATCH_SIZE, shuffle=True)
vqvae_testset = tv.datasets.ImageFolder(PATH + '/keras_png_slices_test', transform=transform)
vqvae_test_loader = t.utils.data.DataLoader(vqvae_testset, batch_size=BATCH_SIZE, shuffle=True)