"""
Created on Monday Sep 18 12:20:00 2023

This script is for loading in the downloaded OASIS Brain data set(9K images), creating a data loader 
and performing any required preprocessing it before training. 
The data is a preprocessed version of the original OASIS Brain dataset provided by COMP3710 course staff.

@author: Gabriel Russell
@ID: s4640776

"""
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
from modules import *

#Initialising global path directories
current_dir = os.getcwd()
OASIS_train_path = current_dir + '\keras_png_slices_train'
OASIS_validate_path = current_dir + '\keras_png_slices_validate'
OASIS_test_path = current_dir + '\keras_png_slices_test'

"""
This class takes the downloaded data from a specific path, performs the required 
transformation to all images and returns it ready to be loaded into a dataloader
"""
class OASISDataset(Dataset):
    #Define the transform as a class attribute
    transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor()
    ])

    def __init__(self, data_directory):
        """
        Initialises attributes for class.

        Args: 
            data_directory (str): Path for data to be imported

        Returns:
            None
        """
        self.data_directory = data_directory
        self.data = os.listdir(data_directory)

    def __len__(self):
        """
        Gets length of data

        Args: 
            None

        Returns:
            Length of data
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Allows instances of the class to be accessed using square bracket notation

        Args: 
            idx (num): Indexed number specified

        Returns:
            img (tensor): Image at specified path after being converted to a tensor
        """
        img_path = os.path.join(self.data_directory, self.data[idx])
        img = Image.open(img_path)
        img = self.transform(img)

        return img

"""
This class calls the OASISDataset Class in it's initialisations to set up 
the train, validation and test datasets.
Included are some getter functions for returning the specified dataloader.
"""
class OASISDataloader():
    def __init__(self):
        """
        Initialises attributes for class.

        Args: 
            None

        Returns:
            None
        """
        p = Parameters()
        self.train = OASISDataset(OASIS_train_path)
        self.validate = OASISDataset(OASIS_validate_path)
        self.test = OASISDataset(OASIS_test_path)
        self.batch_size = p.batch_size
    def get_train(self):
        """
        Retrieves training data

        Args: 
            None

        Returns:
            train_dataloader (DataLoader): Training data loader
        """
        train_dataloader =  DataLoader(self.train, batch_size = self.batch_size, shuffle = True, drop_last= True)
        return train_dataloader
    
    def get_validate(self):
        """
        Retrieves validation data

        Args: 
            None

        Returns:
            validate_dataloader (DataLoader): Validation data loader
        """
        validate_dataloader =  DataLoader(self.validate, batch_size = self.batch_size, shuffle = False)
        return validate_dataloader
    
    def get_test(self):
        """
            Retrieves test data

            Args: 
                None

            Returns:
                test_dataloader (DataLoader): Test data loader
            """
        test_dataloader =  DataLoader(self.test, batch_size = self.batch_size, shuffle = False)
        return test_dataloader
    

"""
This class creates the training data required for the DCGAN. This is 
formed from the encoded images produced by a trained VQVAE model.
"""
class DCGAN_Dataset(Dataset):
    #Define the transform as a class attribute
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    def __init__(self, model):
        """
        Initialises attributes for class.

        Args: 
            model (VQVAEModel): Loaded VQVAE model that has been trained

        Returns:
            None
        """
        self.model = model
        self.image_path = OASIS_train_path
        self.images = os.listdir(OASIS_train_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __len__(self):
        """
        Gets length of data

        Args: 
            None

        Returns:
            Length of data
        """
        return(len(self.images))
    
    def __getitem__(self, idx):
        """
        Returns the encoding indices of images after being passed through 
        a trained VQVAE's encoder, convolutional layer and quantizer

        Args: 
            idx (num): Indexed number specified

        Returns:
            encoding_indices (Tensor): Encoded indices of images
        """
        img_path = os.path.join(self.image_path, self.images[idx])
        img = Image.open(img_path)
        img = self.transform(img)
        img = img.unsqueeze(dim = 0)
        img = img.to(self.device)
        #Encode the training data to serve as inputs for GAN
        VQVAE_encoded = self.model.encoder(img)
        conv = self.model.conv_layer(VQVAE_encoded)
        _,_,_,encoding_indices = self.model.quantizer(conv)
        encoding_indices = encoding_indices.float().to('cuda')
        encoding_indices = encoding_indices.view(64,64) #Reshape to 64x64
        encoding_indices = torch.stack((encoding_indices, encoding_indices, encoding_indices),0)
        return encoding_indices