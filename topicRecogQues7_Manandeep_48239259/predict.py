import os
from PIL import Image
import torchvision.transforms as transforms
from modules import CustomSiameseNetwork
from dataset import CustomDataset
import torchvision.datasets as dset
import torch
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import random
import numpy as np
import PIL.ImageOps
import torch.nn.functional as F
import torch.nn.functional as TorchFun
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torch.autograd import Variable

# Define a function to display an image
def imshow(img, text=None, should_save=False):
    npimg = np.array(Image.open(img))
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic', fontweight='bold',
            bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 10})
    plt.imshow(npimg)
    plt.show()

# Define a function to display a grid of images
def imshow_grid(img, text=None, should_save=False):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic', fontweight='bold',
            bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 10})
    plt.imshow(np.transpose(npimg, (1, 2, 0))
    plt.show()

# Create an instance of the trained Siamese network
trained_siamese_net = CustomSiameseNetwork()
trained_siamese_net.load_state_dict(torch.load('/content/drive/MyDrive/dataset/model.pth'))
trained_siamese_net.eval()

# Define the test dataset
testing_dir = '/content/AD_NC/test'
folder_dataset_test = dset.ImageFolder(root=testing_dir)
siamese_dataset = CustomDataset(folder_dataset_test,
                                transform=transforms.Compose([transforms.Resize((100, 100)),
                                                              transforms.ToTensor()
                                                              ]),
                                should_invert=False)

# Define the data loader for testing
test_dataloader = DataLoader(siamese_dataset, num_workers=6, batch_size=1, shuffle=True)
dataiter = iter(test_dataloader)
x0, _, _ = next(dataiter)

# Iterate through the test dataset and display pairs of images
for i in range(10):
    _, x1, label2 = next(dataiter)
    print(label2)
    concatenated = torch.cat((x0, x1), 0)

    output1, output2 = trained_siamese_net(Variable(x0), Variable(x1))
    euclidean_distance = F.pairwise_distance(output1, output2)
    imshow_grid(torchvision.utils.make_grid(concatenated), 'Dissimilarity: {:.2f}'.format(euclidean_distance.item()))

# Define a function to classify a test image
def classify_test_image(test_image_path):
    test_image = Image.open(test_image_path)
    test_image = test_image.convert("L")
    test_image = transforms.Compose([transforms.Resize((100, 100)), transforms.ToTensor()])(test_image)
    test_image = test_image.unsqueeze(0)

    reference_images_class1 = ['/content/AD_NC/test/AD/1003730_100.jpeg', '/content/AD_NC/test/AD/1003730_101.jpeg']
    reference_images_class2 = ['/content/AD_NC/test/NC/1182968_100.jpeg', '/content/AD_NC/test/NC/1182968_100.jpeg']

    distances_class1 = []
    distances_class2 = []

    for ref_image_path in reference_images_class1:
        ref_image = Image.open(ref_image_path)
        ref_image = ref_image.convert("L")
        ref_image = transforms.Compose([transforms.Resize((100, 100)), transforms.ToTensor()])(ref_image)
        ref_image = ref_image.unsqueeze(0)
        with torch.no_grad():
            output1, output2 = trained_siamese_net(test_image, ref_image)
            euclidean_distance = TorchFun.pairwise_distance(output1, output2)
            distances_class1.append(euclidean_distance.item())

    for ref_image_path in reference_images_class2:
        ref_image = Image.open(ref_image_path)
        ref_image = ref_image.convert("L")
        ref_image = transforms.Compose([transforms.Resize((100, 100)), transforms.ToTensor()])(ref_image)
        ref_image = ref_image.unsqueeze(0)
        with torch.no_grad():
            output1, output2 = trained_siamese_net(test_image, ref_image)
            euclidean_distance = TorchFun.pairwise_distance(output1, output2)
            distances_class2.append(euclidean_distance.item())

    mean_distance_class1 = sum(distances_class1) / len(distances_class1)
    mean_distance_class2 = sum(distances_class2) / len(distances_class2)

    if mean_distance_class1 < mean_distance_class2:
        return "Has Alzheimer Disease"
    else:
        return "Is Cognitive Normal"

# Test image classification and display
test_image_path = '/content/AD_NC/test/AD/1003730_107.jpeg'
classification = classify_test_image(test_image_path)
imshow(test_image_path)
print(f"Test Image: {classification}")

# Define a function to calculate accuracy
def calculate_accuracy(test_folder_path, true_labels):
    correct = 0
    total = 0

    for label, folder_name in true_labels.items():
        folder_path = os.path.join(test_folder_path, folder_name)
        image_files = os.listdir(folder_path)

        for image_file in image_files:
            image_path = os.path.join(folder_path, image_file)
            classification = classify_test_image(image_path)
            if classification == label:
                correct += 1
            total += 1

    accuracy = (correct / total) * 100
    return accuracy

# Calculate and print accuracy for the test folder and true labels
test_folder_path = '/content/AD_NC/test/'
true_labels = {
    'Class 1': 'AD',
    'Class 2': 'NC',
}

accuracy = calculate_accuracy(test_folder_path, true_labels)
print(f"Accuracy: {accuracy}%")
