import os
import numpy as np
import argparse
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

def load_and_predict(model_path, input_image_dir, output_dir):
    # Load the saved model
    model = keras.models.load_model(model_path, compile=False)

    # Define input image dimensions
    input_height = 512
    input_width = 512

    # Create a list of input image file names
    input_image_files = os.listdir(input_image_dir)

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Loop through input images and make predictions
    for input_image_file in input_image_files:
        # Load and preprocess the input image
        input_image_path = os.path.join(input_image_dir, input_image_file)
        img = image.load_img(input_image_path, target_size=(input_height, input_width))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize the image

        # Make predictions using the model
        prediction = model.predict(img_array)

        # Save the prediction as an image (you can customize this part)
        prediction_image = np.squeeze(prediction, axis=0)  # Remove the batch dimension
        prediction_image = (prediction_image * 255).astype(np.uint8)  # Convert to 8-bit image
        prediction_image_path = os.path.join(output_dir, f"prediction_{input_image_file}")
        plt.imsave(prediction_image_path, prediction_image, cmap='gray')

        print(f"Saved prediction for {input_image_file} to {prediction_image_path}")

    print("Prediction complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run predictions using a saved model.")
    parser.add_argument("model_path", type=str, help="Path to the saved model file")
    parser.add_argument("input_image_dir", type=str, help="Directory containing input images for prediction")
    parser.add_argument("output_dir", type=str, help="Directory to save prediction results")
    args = parser.parse_args()

    load_and_predict(args.model_path, args.input_image_dir, args.output_dir)
