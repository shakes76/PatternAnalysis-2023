from dataset import load_dataset, load_data_classifier
from modules import Modules
import tensorflow as tf
import umap
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

"""
Function that creates an instance of a siamese network
and loads the weights from train.py
"""
def load_base():
    model = Modules().base_network()
    model.load_weights("siamese_model.h5", by_name=True, skip_mismatch=True)
    return model

"""
Function that creates an instance of a classifier network
and loads the weights from train.py
"""
def load_classifier():
    model = Modules().create_classifier()
    model.load_weights("best_classifier_model.h5", by_name=True, skip_mismatch=True)
    return model

def create_umap_plot(embeddings, labels):
    reducer = umap.UMAP()
    umap_embeddings = reducer.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], c=labels, cmap='viridis', s=5)
    plt.colorbar()
    plt.title('UMAP Projection of Embeddings')
    plt.show()

"""
Mainloop to produce projections and accuracy statistics
"""
def main():
    # define the data
    train_generator, test_generator = load_dataset()
    test_anchor, test_labels = load_data_classifier(test_generator, 250) # might need to change
    test_labels = tf.argmax(test_labels, axis=1).numpy()

    # create first layer model and create embeddings
    base_model = load_base()
    test_embeddings = base_model.predict(test_anchor)
    create_umap_plot(test_embeddings, test_labels)

    # create the classifier and predict from those embeddings
    classifier = load_classifier()
    test_predictions = classifier.predict(test_embeddings)

    # conver class propabilities to class labels
    predictions = tf.argmax(test_predictions, axis=1).numpy()
    create_umap_plot(test_predictions, test_labels)

    # Calculate accuracy for the training and test sets
    accuracy = accuracy_score(test_labels, predictions)

    print(f"Test Accuracy: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()