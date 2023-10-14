import matplotlib.pyplot as plt
from modules import plot_data
from modules import siamese_network
from modules import classification_model
from modules import contrastive_loss
from dataset import load_siamese_data
from dataset import load_classify_data
from tensorflow.keras.models import load_model



def train_and_plot_SNN(epochs=10):
    # Load Siamese model data
    siamese_train, siamese_val = load_siamese_data()

    # Create Siamese model
    model = siamese_network(128, 128)

    # Train the model
    siamese_fit = model.fit(siamese_train, epochs=epochs, validation_data=siamese_val)

    # Plot the fit data
    plt.figure()
    plt.plot(siamese_fit.history['accuracy'], label='accuracy')
    plt.plot(siamese_fit.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    
    plt.figure()
    plt.plot(siamese_fit.history['loss'], label='loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim([0, 50])
    plt.legend(loc='lower right')

    plt.show()

    return model, siamese_fit