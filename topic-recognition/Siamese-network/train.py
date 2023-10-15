import matplotlib.pyplot as plt
from modules import siamese_network
from modules import contrastive_loss
from modules import classification_model
from dataset import load_train_data
from tensorflow.keras.models import load_model

save_trained_snn = '/Users/jollylogan/TryTry/SNN.h5'
save_trained_classifier = '/Users/jollylogan/TryTry/Classifier.h5'

def train_and_plot_SNN(epochs=10):
    # Load Siamese model data
    siamese_train, siamese_val = load_train_data()

    # Create Siamese model
    model = siamese_network(128, 128)

    # Train the model
    siamese_fit = model.fit(siamese_train, epochs=epochs, validation_data=siamese_val)
    model.save(save_trained_snn)

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


def train_and_plot_classifier(epochs=10):
    # Load classification model data
    classify_train, classify_val = load_train_data(testing=False)

    # Load the trained Siamese model
    siamese_model = load_model(save_trained_snn, custom_objects={'contrastive_loss': contrastive_loss})
    
    # Get the CNN layer from the siamese model
    cnn = siamese_model.get_layer(name="cnn")

    # Create classification model using the subnet
    classifier = classification_model(cnn)

    # Train the classification model
    classifier_fit = classifier.fit(classify_train, epochs=epochs, validation_data=classify_val)
    classifier.save(save_trained_classifier)

    # Plot the fit data
    plt.figure()
    plt.plot(classifier_fit.history['accuracy'], label='accuracy')
    plt.plot(classifier_fit.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')

    plt.figure()
    plt.plot(classifier_fit.history['loss'], label='loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim([0, 1]) 
    plt.legend(loc='lower right')

    plt.show()

    return classifier, classifier_fit


def train():
    siamese_model, siamese_fit = train_and_plot_SNN()
    classifier_model, classifier_fit = train_and_plot_classifier()
        
train()