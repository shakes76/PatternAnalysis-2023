from dataset import load_dataset, create_triplets
from modules import SiameseModel
from train import train
import tensorflow as tf

def load_classifier():
    model = SiameseModel().create_classifier()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Load weights from the Siamese network up to 'dense_2'
    model.load_weights("ADNI_predictor.keras", by_name=True, skip_mismatch=True)

    return model

def main():
    train_generator, test_generator = load_dataset()
    test_triplets, test_labels = create_triplets(test_generator, 250)
    classifier = load_classifier()

    correct_predictions = 0
    total_predictions = len(test_triplets[0])

    # Predict individual images
    predictions = classifier.predict(test_triplets[0])
    predictions = tf.math.argmax(predictions, axis=1).numpy()

    for i in range(len(predictions)):
        print(f"Image {i+1}")
        print(f"Prediction: {predictions[i]}")
        print(f"True Label: {test_labels[0][i]}\n")

        if predictions[i] == test_labels[0][i]:
            correct_predictions += 1

    accuracy = (correct_predictions / total_predictions) * 100.0
    print(correct_predictions)
    print(f"Accuracy: {accuracy:.2f}%")



if __name__=="__main__":
    main()