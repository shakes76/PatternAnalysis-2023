from modules import TripletNet, TripletNetClassifier
import torch
import numpy as np
from dataset import TripletImageTestFolder, get_datasets, get_data
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def predict(tripleClassifier = TripletNetClassifier(), 
                            classifier_load_dir = 'models/tripleClassifier_4.pth', 
                            tripleNet = TripletNet(), 
                            model_load_dir = 'models/model_S7_v3.pth',
                            X = 'AD_NC/test'):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load in data
    test_transform = transforms.Compose([
            transforms.Resize((100, 100)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Grayscale(),
        ])
    
    dataset = get_data(X, transform=test_transform)
    loader = DataLoader(dataset, batch_size=len(dataset))

    # Load the Triplet network
    tripleClassifier.to(device)
    tripleClassifier.load_state_dict(torch.load(classifier_load_dir))

    tripleNet.to(device)
    tripleNet.load_state_dict(torch.load(model_load_dir))

    tripleClassifier.eval()  # Set the model to evaluation mode

    def extract_features(data):
        with torch.no_grad():
            embeddings = tripleNet.forward_one(data)
            return embeddings

    with torch.no_grad():
        for batch in loader:
            test_X, test_y = batch
            test_X, test_y = test_X.to(device), test_y.to(device)
            

            # Forward pass
            features = extract_features(test_X)
            test_outputs = tripleClassifier(features)
            _, predicted = torch.max(test_outputs, 1)
    return predicted

    
    

def accuracyTripletModel(model=TripletNet(), 
                         load_dir = 'models/model_S5_5_v3.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device: ', device)

    train_transform = transforms.Compose([
            transforms.Resize((100, 100)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomCrop(100, padding=4, padding_mode='reflect'),
            transforms.Grayscale(),
        ])
    test_transform = transforms.Compose([
            transforms.Resize((100, 100)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Grayscale(),
        ])

    batch_size = 32

    test_folder = 'AD_NC/test'

    test_dataset = TripletImageTestFolder(test_folder, transform=test_transform)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    model.to(device)
    model.load_state_dict(torch.load(load_dir))

    num_correct = 0
    num_total = 0
    print('Triplet Network accuracy')
    with torch.no_grad():
        for i, images in enumerate(test_loader):
            anchor_image, positive_image, negative_image = images
            anchor_image, positive_image, negative_image = anchor_image.to(device), positive_image.to(device), negative_image.to(device)
            
            # Calculate the distances between the embedded images
            anchor_embedding, positive_embedding, negative_embedding = model(anchor_image, positive_image, negative_image)

            anchor_positive_distance = torch.nn.functional.pairwise_distance(anchor_embedding, positive_embedding)
            anchor_negative_distance = torch.nn.functional.pairwise_distance(anchor_embedding, negative_embedding)
            # If the anchor-positive distance is smaller than the anchor-negative distance, then the triplet Siamese network has correctly classified the triplet
            for i in range(len(anchor_positive_distance)):
                num_total += 1
                if anchor_positive_distance[i] < anchor_negative_distance[i]:
                    num_correct += 1

    accuracy = num_correct / num_total
    print(f'Test Accuracy: {int(accuracy*100)}%')

def accuracyClassifierModel(tripleClassifier = TripletNetClassifier(), 
                            classifier_load_dir = 'models/tripleClassifier_4.pth', 
                            tripleNet = TripletNet(), 
                            model_load_dir = 'models/model_S7_v3.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device: ', device)


    test_transform = transforms.Compose([
            transforms.Resize((100, 100)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Grayscale(),
        ])

    batch_size = 32

    _, test = get_datasets('AD_NC', test_transform)

    test_loader = DataLoader(test, batch_size=batch_size, shuffle=True)


    # Load the Triplet network
    tripleClassifier.to(device)
    tripleClassifier.load_state_dict(torch.load(classifier_load_dir))

    tripleNet.to(device)
    tripleNet.load_state_dict(torch.load(model_load_dir))

    def extract_features(data):
        with torch.no_grad():
            embeddings = tripleNet.forward_one(data)
            return embeddings

    tripleClassifier.eval()  # Set the model to evaluation mode
    with torch.no_grad():

        correct = 0
        total = 0

        # Initialize a dictionary to store class-wise accuracy
        class_correct = {i: 0 for i in range(2)}
        class_total = {i: 0 for i in range(2)}

        for batch in test_loader:
            test_X, test_y = batch
            test_X, test_y = test_X.to(device), test_y.to(device)

            # Forward pass
            features = extract_features(test_X)
            test_outputs = tripleClassifier(features)
            _, predicted = torch.max(test_outputs, 1)
            # Compute overall accuracy
            correct += (predicted == test_y).sum().item()
            total += test_y.size(0)
            # Compute class-wise accuracy
            for i in range(2):
                class_total[i] += (test_y == i).sum().item()
                class_correct[i] += (predicted == i)[test_y == i].sum().item()

        overall_accuracy = correct / total
        print(f"Overall Test Accuracy: {overall_accuracy:.4f}")

        # Print class-wise accuracy
        for i in range(2):
            class_accuracy = class_correct[i] / class_total[i]
            print(f"Class {i} Accuracy: {class_accuracy:.4f}")