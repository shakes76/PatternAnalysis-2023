import os

trained_model = SiameseNetwork()
trained_model.load_state_dict(torch.load('/content/drive/MyDrive/siamese_network.pth'))
trained_model.eval()

# Function to classify a test image based on Euclidean distance
def classify_image(test_image_path):
    # Load and preprocess the test image
    test_image = Image.open('/content/drive/MyDrive/test/melanoma/ISIC_0000056.jpg')
    test_image = test_image.convert("L")
    test_image = transforms.Compose([transforms.Resize((100, 100)), transforms.ToTensor()])(test_image)
    test_image = test_image.unsqueeze(0)  # Add batch dimension

    # Prepare reference images from both classes (adjust paths accordingly)
    reference_images_class1 = ['/content/drive/MyDrive/training/melanoma/ISIC_0000139.jpg', '/content/drive/MyDrive/training/melanoma/ISIC_0000141.jpg']  # List of reference images
    reference_images_class2 = ['/content/drive/MyDrive/training/vascular lesion/ISIC_0024475.jpg', '/content/drive/MyDrive/training/vascular lesion/ISIC_0024662.jpg']  # List of reference images

    # Compute Euclidean distances
    distances_class1 = []
    distances_class2 = []

    for ref_image_path in reference_images_class1:
        ref_image = Image.open(ref_image_path)
        ref_image = ref_image.convert("L")
        ref_image = transforms.Compose([transforms.Resize((100, 100)), transforms.ToTensor()])(ref_image)
        ref_image = ref_image.unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            output1, output2 = trained_model(test_image, ref_image)
            euclidean_distance = F.pairwise_distance(output1, output2)
            distances_class1.append(euclidean_distance.item())