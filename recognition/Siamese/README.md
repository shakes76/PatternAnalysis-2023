# Siamese model

## This project was created by Nathan Levu student id s4682374

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

Before we begin it should be noted that the data was shrunk. This was to make training faster and easier, but also because Siamese networks have the ability to learn with smaller amount's of data.

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

The next important aspect is the Siamese model itself. The Siamese model is set up like other CNN model's but with one key difference. That difference is that the Siamese model takes two images and run's them through the convulation layers before outputing them. Thus the model need's to run two images simultaneously through the model to get two different outputs. The forward function that handles this looks like 
```
    def forward(self, x1, x2):
        output1 = self.features(x1)
        output2 = self.features(x2)
        output1 = output1.view(output1.size(0), -1)
        output1 = self.classifier(output1)
        output2 = output2.view(output2.size(0), -1)
        output2 = self.classifier(output2)
        return output1, output2
```

The reason two output's are given is to make use of the custom contrastive loss function used to train the model. The contrastive loss function take's two image outputs from the model and compares them based on the label. If the label say's the images are from the same class, the loss function calculates the euclidean distance of the two outputs in order to calculate dissimilarity of the two images. On the other hand if the two images are from different classes, the loss function calculates similarity of the two images. This is so that images in the same class can be as similar as possible while images in different classes can be as different as possible. 

### The loss contrastive function used look's like this 
```
class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, img1, img2, label):
        distance = torch.nn.functional.pairwise_distance(img1, img2, keepdim=True) #distance finds similarities between images
        reverse_dist = self.margin - distance #reverse distance find's difference between images
        reverse_dist = torch.clamp(reverse_dist, min=0.0)
        #if label is 1, returns distance, else if label is 0 returns reverse distance
        return torch.mean(torch.pow(reverse_dist, 2) * (abs(label-1)) + ((torch.pow(distance, 2)) * label))
```

Now on to training the model. In order to train the model, you will need to load two image's and their label. Train the model using those two images and take the output and the given label and run it through the loss function and backpropagate it. This will help train the model and hopefully improve it. With the use of the contrastive loss function, differences between two different classes should be highlightsed more and similarities between the same class should be highlighted more.

### Example usage of train function
```
layer = "VGG16"
    in_channels = 1
    classes = 2
    epochs = 5
    learning_rate = 1e-5  

    model = md.Siamese(layers=layer, in_channels=in_channels, classes=classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

    tr.model_train(model, optimizer, epochs)
```

To test the model, we want to test it's ability to find disimilarity between two images. We give the model two images and take the output and find the disimilarity. If disimilarity is below a certain threshold, say 0.5 we can assume their the same class. That is the model's prediction. Next we compare that to the actual label to see if the outcome was correct.
```
    model.eval()
    print("begin testing")
    correct = 0
    total = 0
    for (img1, img2, label) in ds.testloader:
        img1 = img1.to(device)
        img2 = img2.to(device)
        label = label.to(device)
        output1, output2 = model(img1, img2)
        distance = torch.nn.functional.pairwise_distance(output1, output2)
        pred = torch.where(distance > 0.5, 0.0, 1.0)
        right = torch.where(label == pred, 1, 0)
        guesses = right.size(dim=0)
        total = total + guesses
        correct = correct + torch.sum(right).item()
```

As you can see from the code, pairwise distance is used to find the dissimilarity. 

### Example usage

Now we run some of our code and see how it does. First ill run the training and plot the loss function to see if our model is actuall learning. 

![Figure_3](https://github.com/Picayune1/PatternAnalysis-2023/assets/141021565/9f5e6bb1-5d52-4d7a-b90f-96525485fc9f)

As you can see the loss does seem to show a steady trend of decreasing. The model used to generate this example is below and was given about 2.5 hours to train through the dataset
```
layer = "VGG16"
    in_channels = 1
    classes = 2
    epochs = 5
    learning_rate = 1e-5  

    model = md.Siamese(layers=layer, in_channels=in_channels, classes=classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
```

The accuracy produced was 60% which may be due to learning, or may be due to random chance. I do beleive though that the model and loss function used were correct, and if better image preprocessing was used and the model had more time to tran and ran through more epochs, it could produce and accuracy of 80%.



