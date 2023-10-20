"""
ds.py
Author: Francesca Brzoskowski
s4705512
Contains the data loader for loading and preprocessing for OASIS data
"""

import os
import sys
import zipfile

import requests
import numpy as np
import tensorflow as tf

class Data:
    """
    A class to represent a Data model.

    ...

    Methods
    -------
    get_data(self):
        Downloads dataset
    unzip(self):
        Unzips dataset
    load_data(self, folder_name: str):
        Unzips and loads data in a dataloader
    scale_image(self, image):
        Scales image to 128x128
    preprocess_data(self, dataset):
        Preprocesses data by scaling each image in dataset
    get_test_dataset(self):
        gets the testing dataset
    get_test_dataset(self):
        gets the testing dataset
    
    """
    def __init__(self):
        """
        Constructs all the necessary attributes for the Data model.
        
        """
        self.ds_location = "https://cloudstor.aarnet.edu.au/plus/s/tByzSZzvvVh0hZA/download"
        self.ds = "ds/"
        self.ds_zip_name = "data.zip"
        self.ds_name = "keras_png_slices_data/"
        self.ds_train = "keras_png_slices_train"
        self.ds_test = "keras_png_slices_test"
        self.ds_val = "keras_png_slices_validate"

        self.image_size = (128, 128)
        
    def get_data(self):
        ''' Downloads dataset
            Parameters:
                idx: image index
            
            Returns:
                None
        '''
        if not os.path.isdir(self.ds):
            os.mkdir(self.ds)

        # Download dataset zip
        if not os.path.exists(self.ds + self.ds_zip_name):
            print(f"Downloading dataset into ./{self.ds}{self.ds_zip_name}")
            response = requests.get(self.ds_location, stream=True)
            total_length = response.headers.get("content-length")

            with open(self.ds + self.ds_zip_name, "wb") as f:
                # Show download progress bar (doesn't work on non-Unix systems)
                # Adapted from https://stackoverflow.com/a/15645088
                if total_length is None or not os.name == "posix":
                    f.write(response.content)
                else:
                    dl = 0
                    total_length = int(total_length)
                    for data in response.iter_content(chunk_size=4096):
                        dl += len(data)
                        f.write(data)
                        done = int(50 * dl / total_length)
                        sys.stdout.write("\r[%s%s]" % ('=' * done, ' ' * (50-done)))
                        sys.stdout.write(" %d%%" % int(dl / total_length * 100))
                        sys.stdout.flush()
                

            print("Dataset downloaded.\n")
        #else:
            #print("Dataset already downloaded.\n")

    def unzip(self):
        """ Unzips dataset
            Returns:
                None
        """
        # unzip 
        if not os.path.isdir(self.ds + self.ds_name):
            print(f"Extracting dataset into ./{self.ds}{self.ds_name}")
        with zipfile.ZipFile(self.ds + self.ds_zip_name) as z:
            z.extractall(path=self.ds)
        print("Dataset extracted.\n")


    def load_data(self,folder_name: str):
        """ Loads data in a dataloader
            Parameters:
                folder_name: name of folder to load from
            Returns: 
                OASIS dataset
        """
        

        # load
        ds = tf.keras.preprocessing.image_dataset_from_directory(
        self.ds + self.ds_name + folder_name,
        labels=None,
        image_size=self.image_size,
        )

        return ds

    def scale_image(self, image: tf.Tensor) -> tf.Tensor:
        """ Scales image to 128x128
            Parameters:
                image: image from dataset
            Returns: 
            rescaled image
        """
        image = image / 255 - 0.5
        image = tf.image.rgb_to_grayscale(image)
        return image

    
    def preprocess_data(self, dataset: tf.data.Dataset) -> np.array:
        """ Preprosses data by scaling each image in dataset
        Parameters:
            datast: the OASIS dataset
        Returns: an array of data
        """
        return np.asarray(list(dataset.unbatch().map(self.scale_image)))
    
    def get_train_dataset(self) -> np.array:
        """ gets the training dataset
        Parameters:
            folder_name: name of training folder
        Returns:
            training dataset
        """

        self.get_data()
        self.unzip()

        train_ds = self.load_data(self.ds_train)
        train_ds = self.preprocess_data(train_ds)

        return train_ds

    def get_test_dataset(self) -> np.array:
        """ gets the testing dataset
        Parameters:
            folder_name: name of test folder
        Returns:
            testing dataset
        """
        self.get_data()
        self.unzip()

        test_ds = self.load_data(self.ds_test)
        test_ds = self.preprocess_data(test_ds)

        return test_ds


# if __name__ == "__main__":
#      ds = Data()
#      train = ds.get_train_dataset()
#      test = ds.get_test_dataset()


#      print("train", "size", train.size, "shape", train.shape)
#      print("test", "size", test.size, "shape", test.shape)
