import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
from datasets import adniDataset, embeddingsDataset
from model import Siamese, Classifier
from train import train_encoder, train_classifier, validate_model, test_model

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.__version__)
print(device)
print(torch.cuda.device_count())

# Path
path = '/home/groups/comp3710'
tensor_path = '/home/Student/s4623300/PatternAnalysis-2023/recognition/46233002_Siamese_ADNI/easysemihard'
model_path = '/home/Student/s4623300/PatternAnalysis-2023/recognition/46233002_Siamese_ADNI/models'

# Hyper-parameters
num_epochs_encoder = 24
num_epochs_classifier = 24
learning_rate_encoder = 1e-04
learning_rate_classifier = 1e-03
batch_size = 16

# Get file names
train_ads = os.listdir(path + "/ADNI/AD_NC/train/AD") # len = 10400
train_ncs = os.listdir(path + "/ADNI/AD_NC/train/NC") # len = 11120
test_ads = os.listdir(path + "/ADNI/AD_NC/test/AD")   # len = 4460
test_ncs = os.listdir(path + "/ADNI/AD_NC/test/NC")   # len = 4540

# Extract validation set from test set 
val_size = 0.1
val_ads = train_ads[:int(len(train_ads)*val_size)]
val_ncs = train_ncs[:int(len(train_ncs)*val_size)]
train_ads = train_ads[int(len(train_ads)*val_size):]
train_ncs = train_ncs[int(len(train_ncs)*val_size):]

# Check data distribution
print("\n === Hyperparameters of Models === ")
print("Number of Epochs (Encoder/Classifier): ", num_epochs_encoder, num_epochs_classifier)
print("Learning Rate: ", learning_rate_encoder, learning_rate_classifier)
print("Batch Size: ", batch_size)
print("\n ==== Distribution of ads/ncs ==== ")
print("Train set: ", len(train_ads), len(train_ncs))  # 8320, 8896
print("Test set: ", len(test_ads), len(test_ncs))     # 4460, 4540
print("Validation set: ", len(val_ads), len(val_ncs)) # 2080, 2224
print()

# Transformer
train_transform = transforms.Compose([transforms.ToTensor(),
                                transforms.RandomCrop((225, 225)),
                                transforms.RandomAffine(degrees=25, scale=(0.9, 1.1)),
                                transforms.Normalize(mean=0.0, std=1.0)])

val_transform = transforms.Compose([transforms.ToTensor(),
                                transforms.CenterCrop((240, 240)),
                                transforms.Resize((225, 225)),
                                transforms.Normalize(mean=0.0, std=1.0)])

test_transform = transforms.Compose([transforms.ToTensor(),
                                transforms.CenterCrop((240, 240)),
                                transforms.Resize((225, 225)),
                                transforms.Normalize(mean=0.0, std=1.0)])

# Create train/test/validation dataset objects 
trainset = adniDataset(ad_dir = path + "/ADNI/AD_NC/train/AD", 
                       nc_dir = path + "/ADNI/AD_NC/train/NC",
                       ads = train_ads, ncs = train_ncs,
                       transform = train_transform)
testset = adniDataset(ad_dir = path + "/ADNI/AD_NC/test/AD", 
                      nc_dir = path + "/ADNI/AD_NC/test/NC",
                      ads = test_ads, ncs = test_ncs,
                      transform = test_transform)
valset = adniDataset(ad_dir = path + "/ADNI/AD_NC/train/AD", 
                     nc_dir = path + "/ADNI/AD_NC/train/NC",
                     ads = val_ads, ncs = val_ncs,
                     transform = val_transform)


# Load datasets to DataLoader 
train_loader = DataLoader(trainset, shuffle=True, batch_size=batch_size)
val_loader = DataLoader(valset, shuffle=False, batch_size=batch_size)
test_loader = DataLoader(testset, shuffle=False, batch_size=batch_size)

# Construct encoder 
siamese = Siamese().to(device)
print(siamese)

# Train encoder
train_encoder(siamese, train_loader, device, num_epochs_encoder, tensor_path, 
              learning_rate_encoder)
#torch.save(siamese.state_dict(), model_path + '/94168.pt')

# Get embeddings
embeddings_set = embeddingsDataset(tensor_path, device)
embeddings_loader = DataLoader(embeddings_set, shuffle=True, batch_size=1)

# Construct classifier
classifier = Classifier().to(device)
print(classifier)

# Train classifier
train_classifier(classifier, embeddings_loader, device, num_epochs_classifier, 
                 learning_rate_classifier)
#torch.save(classifier.state_dict(), model_path + '/94168_classifier.pt')

# Validate models
validate_model(siamese, classifier, val_loader, device)

# Test models 
test_model(siamese, classifier, test_loader, device)
