import os
from keras.callbacks import TensorBoard, ModelCheckpoint
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
    train_triplets, train_labels = create_triplets(train_generator, 1200)
    test_triplets, test_labels = create_triplets(test_generator, 300)

    log_dir = os.path.join('Logs')
    tb_callback = TensorBoard(log_dir=log_dir)

    # Define a ModelCheckpoint callback to save the best model based on validation loss
    checkpoint_callback = ModelCheckpoint(
        "best_model.h5",
        monitor='val_loss',  # Monitor validation loss
        save_best_only=True,  # Save only the best model
        mode='min',  # The best model has the lowest validation loss
        verbose=1
    )

    history = model.fit(
        train_triplets,
        train_labels,
        epochs=25,
        batch_size=32,
        validation_data=(test_triplets, test_labels),
        callbacks=[tb_callback, checkpoint_callback],
    )
    
    model.save("ADNI_predictor.h5")

    # Extract loss values from the training history
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']

    # Plot the loss values
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Over Epochs')
    plt.show()

def main():
    train()



if __name__=="__main__":
    main()



