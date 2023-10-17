# Siamese model

### Description
The Siamese model is a neural network that work's on two given input's to create comparable outputs. These output's can be used to judge the similarities between given inputs. Being able to compare the similarities of two input's is very useful, and Siamese network's can be used for:
- Facial recognition
- Signature analysis
- Matching queries with documents

This Siamese network is used to solve the issue of identifying Alzheimers within the brain. Using the dataset provided by the **ADNI** (Alzheimers disease neuroimaging initiative). It take's an image from one of the two classes, normal and Alzheimers disease, Then selects a 2nd image from one of the two classes and attempts to find if they are from the same class or not. By learning features in both a normal brain and one with Alzheimers disease, it can learn to spot certain similarities and differences between the two classes. This can then hopefully be used to train a model that can produce an accuracy of 80%

### How the algorithm works
Before we jump into how the algorithm works and some of the components in the model, first I will list imported and downloaded modules and libraries:
- Pytorch (version 2.1)
- Cuda (version 11.8)
- Numpy (version 1.24.0)
- data (link : https://cloudstor.aarnet.edu.au/plus/s/L6bbssKhUoUdTSI)
- Matplotlib (version 3.7)

**Make sure to use these or newer versions of these libraries**

Now to get started the first important feature of a Siamese network is a custom dataset. This Siamese network take's two images from the dataset to compare, and thus the dataset should return two images as well as a label. The label is not the class of image 1 or 2 but instead informs if the two images are from the same class. Let's take a look at the code
```
class Siamese_dataset(Dataset):
    def __init__(self, imageFolder, transform):
        self.imageFolder = imageFolder
        self.transform = transform
    
    def __len__(self):
        return len(self.imageFolder.imgs)

    def __getitem__(self, index):
        #we need two images in order to compare similarity
        img0 = random.choice(self.imageFolder.imgs)

        #First we decide if were selecting two images with different classes or the same
        same_class = random.randint(0,1)

        if same_class: #if 1 then select two images with the same class
            while True:
                img1 = random.choice(self.imageFolder.imgs)
                if img0[1] == img1[1]:
                    break
        
        while True:
            img1 = random.choice(self.imageFolder.imgs)
            if img0[1] != img1[1]:
                break
        
        img0_Image = Image.open(img0[0])
        img1_Image = Image.open(img1[0])

        img0_Image = img0_Image.convert("L")
        img1_Image = img1_Image.convert("L")

        img0_Image = self.transform(img0_Image)
        img1_Image = self.transform(img1_Image)

        return img0_Image, img1_Image, torch.tensor(same_class, dtype=torch.float)
   ```

The important part of this datset is the getitem function. It's the reason for needing a custom dataset, since we want to grab two images instead of 1 which is how the standard dataset class works. It grabs 1 image and then performs a random selection between 0 and 1. This is to balance the use of comparing images in the same class and different classes. If 1, then the second image will be in the same class while a 0 indicates the second image is in a different class. It then turn's both images into the image data type and returns them along with the label.

Here is an example of using the custom dataset
```
train_path = "C:\\Users\\Asus\\Desktop\\AD_NC\\train"
training_dataset = datasets.ImageFolder(root=train_path)
transform = transforms.Compose([transforms.ToTensor()])
siamese_train = Siamese_dataset(imageFolder=training_dataset, transform=transform)
loader = DataLoader(ds.siamese_train,
                        shuffle=True,
                        batch_size=8)
    
    image1, image2, label = next(iter(loader))
    concatenated = torch.cat((image1, image2),0)

    img = torchvision.utils.make_grid(concatenated).numpy()
    plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.show()
    print(label.numpy().reshape(-1))
```

The output is 

![Figure_1](https://github.com/Picayune1/PatternAnalysis-2023/assets/141021565/c7f0a51c-d71f-42d1-9bbe-66ba8e5d902d)

With a label tensor showing which two images are from the same class 

[0. 1. 0. 1. 1. 0. 0. 0.]

The next important aspect is the 
