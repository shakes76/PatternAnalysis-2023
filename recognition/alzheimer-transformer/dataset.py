'''
contains the data loader for loading and preprocessing the data
'''
from torchvision.datasets import ImageFolder
from torchvision import transforms
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

'''
Returns a list of unique patient IDs given the dataset path
'''
def get_patient_ids(data_path):
    # Knowing that patient IDs are encoded in image filenames as 'patientID_xx.jpeg'
    all_files = [os.path.basename(f) for dp, dn, filenames in os.walk(data_path) for f in filenames]
    patient_ids = list(set([f.split('_')[0] for f in all_files]))
    return patient_ids

'''
Returns the train, val, and test dataloaders for the AD_NC dataset, with a train/val split of 80/20
'''
def get_alzheimer_dataloader(batch_size:int=32, img_size:int=224, path:str="./dataset/AD_NC"):
    # Paths to training and test datasets
    train_data_path = path+"/train"
    test_data_path = path+"/test"

    # Transformers 
    train_transforms = transforms.Compose([
    transforms.Resize((img_size, img_size)),  # Resize to a consistent size
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
    ])

    test_transforms = transforms.Compose([
    transforms.Resize((img_size, img_size)), 
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])

    # Get patient ids from training data path and split these with 80/20
    patient_ids = get_patient_ids(train_data_path)
    train_patient_ids, val_patient_ids = train_test_split(patient_ids, test_size=0.2, random_state=42)

    # Create datasets
    train_dataset = ImageFolder(root=train_data_path, transform=train_transforms)
    test_dataset = ImageFolder(root=test_data_path, transform=test_transforms)

    # Map ID split to the images to get patient-level split
    train_indices = [i for i, (path, label) in enumerate(train_dataset.samples) if path.split('/')[-1].split('_')[0] in train_patient_ids]
    val_indices = [i for i, (path, label) in enumerate(train_dataset.samples) if path.split('/')[-1].split('_')[0] in val_patient_ids]

    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(train_dataset, val_indices)

    # Create data loaders
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader, test_loader


    