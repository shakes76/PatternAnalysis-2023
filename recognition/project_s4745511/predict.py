from tensorflow.keras.models import load_model
from dataset import classification_data_loader

# Define the path to the saved classifier model
CLASSIFIER_PATH = '/content/Classifier.h5'

def prediction():
    # Load the trained classifier model
    classifier = load_model(CLASSIFIER_PATH)

    # Load the test data for classification
    classify_test_data = classification_data_loader(testing=True)

    # Evaluate the classifier on the test data
    evaluation_results = classifier.evaluate(classify_test_data)

    # Display evaluation results
    print("Evaluation Results:")
    print("Loss:", evaluation_results[0])
    print("Accuracy:", evaluation_results[1])

    for pair, label in classify_test_data:
        # Make predictions using the classifier
        predictions = classifier.predict(pair)

        # Display predicted values and actual labels for the first batch
        for i in range(len(predictions)):
            print("Prediction:", predictions[i])
            print("Actual Label:", label[i])

        break  # Stop after the first batch of predictions

# Run the prediction function
prediction()
