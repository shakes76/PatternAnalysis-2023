import os
from keras.callbacks import TensorBoard, ModelCheckpoint
from dataset import load_dataset, load_data_triplets, load_data_classifier
from modules import Modules
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import tensorflow as tf

"""
Function used to train a siamese neural network with the 
data from dataset.py. The model comes from modules.py.

Also plots the loss curve for training and validation set

Inputs: 
    train_triplets - (anchor, positive, negative) train data
    test_triplets - (anchor, positive, negative) test data
    train_labels - (anchor, positive, negative) train labels
    test_labels - (anchor, positive, negative) test labels
"""
def train_siamese(train_triplets, test_triplets, train_labels, test_labels):
    siamese_model = Modules().create_siamese_network()
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    log_dir = os.path.join('Logs')
    tb_callback = TensorBoard(log_dir=log_dir)

    # train the model and save the history
    history = siamese_model.fit(
        train_triplets, 
        train_labels, 
        epochs=25, 
        batch_size=32,
        validation_data=(test_triplets, test_labels), 
        callbacks=[tb_callback, early_stopping]
    )

    siamese_model.save_weights("siamese_model.h5")
    
    # extract loss information for training set
    train_anchor_loss = history.history['model_loss']
    train_positive_loss = history.history['model_1_loss']
    train_negative_loss = history.history['model_2_loss']

    # Plot the loss values for anchor branch
    plt.plot(train_anchor_loss, label='Training Anchor Loss')
    plt.plot(train_positive_loss, label='Training Positive Loss')
    plt.plot(train_negative_loss, label='Training Negative Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Loss vs Epochs')
    plt.show()

    # extract loss information for validation set
    val_anchor_loss = history.history['val_model_loss']
    val_positive_loss = history.history['val_model_1_loss']
    val_negative_loss = history.history['val_model_2_loss']

    # Plot the loss values for positive branch
    plt.plot(val_anchor_loss, label='Validation Anchor Loss')
    plt.plot(val_positive_loss, label='Validation Positive Loss')
    plt.plot(val_negative_loss, label='Validation Negative Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Testing Loss vs Epochs')
    plt.show()

"""
Funtion used to train a classifier based upon the network embeddings
created when training. 

Also displays a training and validation accuracy plot. 

Inputs: 
    train_triplets - (anchor, positive, negative) train data
    test_triplets - (anchor, positive, negative) test data
    train_labels - (anchor, positive, negative) train labels
    test_labels - (anchor, positive, negative) test labels
"""
def train_classifier(train_data, test_data, train_labels, test_labels):
    siamese_model = Modules().base_network()
    siamese_model.load_weights("siamese_model.h5", by_name=True, skip_mismatch=True)
    train_embeddings = siamese_model.predict(train_data)
    test_embeddings = siamese_model.predict(test_data)

    classifier_model = Modules().create_classifier()
    log_dir = os.path.join('Logs')
    tb_callback = TensorBoard(log_dir=log_dir)

    # save best weight based on min val_loss
    checkpoint_callback = ModelCheckpoint(
        "best_classifier_model.h5", 
        monitor='val_loss', 
        save_best_only=True, 
        mode='min', 
        verbose=1
    )

    # store history in a callback for plots
    history = classifier_model.fit(
        train_embeddings,
        train_labels,
        epochs=1000,
        batch_size=32,
        validation_data=(test_embeddings, test_labels),
        callbacks=[tb_callback, checkpoint_callback],
    )

    # Plot training & validation accuracy values
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

"""
Mainloop used for training the train and test set. 
"""
def main():
    train_generator, test_generator = load_dataset()
    train_triplets, train_labels = load_data_triplets(train_generator, 1000) # change depending on system
    test_triplets, test_labels = load_data_triplets(test_generator, 250)
    train_siamese(train_triplets, test_triplets, train_labels, test_labels)

    train_data, train_labels = load_data_classifier(train_generator, 1000) # change depending on system
    test_data, test_labels = load_data_classifier(test_generator, 250)
    train_classifier(train_data, test_data, train_labels, test_labels)

if __name__=="__main__":
    main()