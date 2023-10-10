import os
from keras.callbacks import TensorBoard
from dataset import load_dataset
from modules import SiameseModel
import matplotlib.pyplot as plt

"""
Function used for training the siamese network

The data comes from dataset.py and the model comes from modules.py
"""
def train():
    model = SiameseModel().model  # load the model
    train_generator, test_generator = load_dataset() # load the data

    # fit the model to the data
    model.fit(train_generator, epochs=10, 
                        validation_data=test_generator)
    
    # Save the entire model
    model.save("ADNI_predictor.keras")



def main():
    train()

if __name__=="__main__":
    main()
