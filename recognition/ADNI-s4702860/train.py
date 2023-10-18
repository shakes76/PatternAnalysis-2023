import os
from keras.callbacks import TensorBoard
from dataset import load_dataset, create_triplets
from modules import SiameseModel
import matplotlib.pyplot as plt


"""
Function used for training the siamese network. Plots the loss and accuracy over each epoch. 

The data comes from dataset.py and the model comes from modules.py

https://youtu.be/DGJyh5dK4hU?si=2LxFEx1aOVHJEFBu
"""

def train():
    model = SiameseModel().model
    train_generator, test_generator = load_dataset()
    train_triplets, train_labels = create_triplets(train_generator, 1000)
    test_triplets, test_labels = create_triplets(test_generator, 250)

    log_dir = os.path.join('Logs')
    tb_callback = TensorBoard(log_dir=log_dir)
    history = model.fit(
        train_triplets, train_labels,
        epochs=25,
        batch_size=32,
        validation_data=(test_triplets, test_labels),
        callbacks=[tb_callback]
    )
    
    model.save("ADNI_predictor.keras")

    #plt.figure(figsize=(12, 4))
    #plt.subplot(1, 2, 1)
    #plt.plot(history.history['loss'], label='Training Loss')
    #plt.plot(history.history['val_loss'], label='Validation Loss')
    #plt.xlabel('Epochs')
    #plt.ylabel('Loss')
    #plt.legend()

    #plt.subplot(1, 2, 2)
    #plt.plot(history.history['accuracy'], label='Training Accuracy')
    #plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    #plt.xlabel('Epochs')
    #plt.ylabel('Accuracy')
    #plt.legend()
    #plt.show()

def main():
    train()



if __name__=="__main__":
    main()
