import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from dataset import ADNC_Dataset, get_image_paths_from_directory, extract_patient_id

def configuration():
    # Initialize the model, loss function, and optimizer
    config_params_dict = {
        "general": {
            "num_channels": 3,  
            "num_classes": 2
        },
        "num_classes": 2,
        "patches": {
            "sizes": [7, 3, 3],
            "strides": [4, 2, 2],
            "padding": [2, 1, 1]
        },
        "transformer": {
            "embed_dim": [64, 192, 384],
            "hidden_size": 384,
            "num_heads": [1, 3, 6],  
            "depth": [1, 1, 1],  
            "mlp_ratios": [4.0, 4.0, 4.0, 4.0],
            "attention_drop_rate": [0.0, 0.0, 0.0],
            "drop_rate": [0.0, 0.0, 0.0],
            "drop_path_rate": [0.0, 0.0, 0.1],
            "qkv": {
                "bias": [True, True, True],
                "projection_method": ["dw_bn", "dw_bn", "dw_bn"],
                "kernel": [3, 3, 3],
                "padding": {
                    "kv": [1, 1, 1],
                    "q": [1, 1, 1]
                },
                "stride": {
                    "kv": [2, 2, 2],
                    "q": [1, 1, 1]
                }
            },
            "cls_token": [False, False, True]
        },
        "initialisation": {
            "range": 0.02,
            "layer_norm_eps": 1e-6
        }
    }
    return config_params_dict

def load_data(train_images_paths_AD, train_images_paths_NC):
    """
    Load and split image dataset into training and validation sets, whilst ensuring that there's no patient overlap between the training and validation sets.
    """
    # Get image paths for training and test datasets
    all_train_images_paths_NC = get_image_paths_from_directory(train_images_paths_NC)
    all_train_images_paths_AD = get_image_paths_from_directory(train_images_paths_AD)

    ## Extract unique patient IDs
    all_patient_ids_AD = list(set(extract_patient_id(path) for path in all_train_images_paths_AD))
    all_patient_ids_NC = list(set(extract_patient_id(path) for path in all_train_images_paths_NC))

    # Split patient IDs into training and validation sets (20%, 80% split)
    train_patient_ids_AD, val_patient_ids_AD = train_test_split(all_patient_ids_AD, test_size=0.2, random_state=42) 
    train_patient_ids_NC, val_patient_ids_NC = train_test_split(all_patient_ids_NC, test_size=0.2, random_state=42)

    # train_patient_ids_AD,val_patient_ids_AD=all_patient_ids_AD,all_patient_ids_AD
    # train_patient_ids_NC,val_patient_ids_NC=all_patient_ids_NC,all_patient_ids_NC
    # Map patient IDs back to image paths for training and validation sets
    train_images_AD = [path for path in all_train_images_paths_AD if extract_patient_id(path) in train_patient_ids_AD]
    val_images_AD = [path for path in all_train_images_paths_AD if extract_patient_id(path) in val_patient_ids_AD]
    train_images_NC = [path for path in all_train_images_paths_NC if extract_patient_id(path) in train_patient_ids_NC]
    val_images_NC = [path for path in all_train_images_paths_NC if extract_patient_id(path) in val_patient_ids_NC]

    return train_images_AD, train_images_NC, val_images_AD, val_images_NC

def create_data_loaders(train_images_AD, train_images_NC, val_images_AD, val_images_NC, batch_size):
    """
    Creates dataloaders for training and validation sets.
    """
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    train_dataset = ADNC_Dataset(train_images_AD, train_images_NC, transform=data_transforms['train'])
    val_dataset = ADNC_Dataset(val_images_AD, val_images_NC, transform=data_transforms['val'])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_dataloader, val_dataloader

def load_test_data(test_images_paths_AD, test_images_paths_NC):
    """
    Loads test data from specified directory and filters patiennt ID
    """
    all_test_images_paths_NC = get_image_paths_from_directory(test_images_paths_NC)
    all_test_images_paths_AD = get_image_paths_from_directory(test_images_paths_AD)

    all_patient_ids_AD_test = list(set(extract_patient_id(path) for path in all_test_images_paths_AD))
    all_patient_ids_NC_test = list(set(extract_patient_id(path) for path in all_test_images_paths_NC))

    # Map patient IDs back to image paths for test set
    test_images_AD = [path for path in all_test_images_paths_AD if extract_patient_id(path) in all_patient_ids_AD_test]
    test_images_NC = [path for path in all_test_images_paths_NC if extract_patient_id(path) in all_patient_ids_NC_test]

    return test_images_AD, test_images_NC