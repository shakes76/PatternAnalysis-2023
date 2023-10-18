import argparse
import os
import matplotlib.pyplot as plt
import torch
from dataset import get_test_loader, get_user_data_loader
from modules import ViT
from sklearn.metrics import confusion_matrix, accuracy_score


# Define command-line arguments
parser = argparse.ArgumentParser(description="Predict Alzheimer's Disease from Images")
parser.add_argument("--model_path", required=True, help="Path to model weights file")
parser.add_argument(
    "--image_folder",
    default=None,
    help="Path to the folder containing images for prediction",
)
parser.add_argument(
    "--output_folder",
    default="./predictions",
    help="Path to the folder where prediction results will be saved",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=64,
    help="Batch size for making predictions (if --image_folder is provided)",
)
args = parser.parse_args()

if __name__ == "__main__":
    # Define the base output folder
    output_folder = args.output_folder

    # Check if the specified output folder already exists
    if os.path.exists(output_folder):
        # Find the next available folder name with a number on the end
        i = 1
        while True:
            new_output_folder = f"{args.output_folder}_{i}"
            if not os.path.exists(new_output_folder):
                output_folder = new_output_folder
                break
            i += 1

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Check if the specified model weights file exists
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model weights file '{args.model_path}' not found.")

    # Initialize the model
    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # Load the model
    model = ViT(
        in_channels=3,
        patch_size=14,
        emb_size=768,
        img_size=224,
        depth=14,
        n_classes=2,
    ).to(device)

    # Load trained model weights
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    # Load data loaders based on user input
    if args.image_folder:
        # User provided a folder with images
        data_loader = get_user_data_loader(
            root_dir=args.image_folder, batch_size=args.batch_size
        )
    else:
        # Use the test loader from your dataset module
        data_loader = get_test_loader()

    # Define class labels
    class_labels = ["AD", "NC"]

    # Define the directory for saving images
    images_output_dir = os.path.join(output_folder, "images")
    os.makedirs(images_output_dir, exist_ok=True)

    # Define the path for saving predictions text file
    results_filename = os.path.join(output_folder, "predictions.txt")

    # Create the output folder and subfolder if they don't exist
    os.makedirs(os.path.dirname(results_filename), exist_ok=True)

    results_file = open(results_filename, "w")

    # Lists to store true labels and predicted labels for all batches
    all_true_labels = []
    all_predicted_labels = []

    # Predict classes for images and visualize after each batch
    for batch_idx, (images, labels) in enumerate(data_loader):
        images = images.to(device)
        with torch.no_grad():
            logits = model(images)
        preds = logits.argmax(dim=1)

        # Convert tensors to NumPy arrays for visualization
        images_np = images.cpu().numpy()
        labels_np = labels.cpu().numpy()
        preds_np = preds.cpu().numpy()

        # Convert tensors to NumPy arrays for confusion matrix calculation
        labels_np = labels.cpu().numpy()
        preds_np = preds.cpu().numpy()

        # Append true and predicted labels to the lists
        all_true_labels.extend(labels_np)
        all_predicted_labels.extend(preds_np)

        # Visualize first image with labels
        plt.imshow(images_np[0].transpose((1, 2, 0)))
        plt.title(
            f"Actual: {class_labels[labels_np[0]]}, Predicted: {class_labels[preds_np[0]]}"
        )
        image_filename = f"batch{batch_idx}_image{1}.png"
        image_filepath = os.path.join(images_output_dir, image_filename)
        plt.savefig(image_filepath)
        plt.close()

    # Compute the confusion matrix
    conf_matrix = confusion_matrix(all_true_labels, all_predicted_labels)

    # Compute the overall accuracy
    accuracy = accuracy_score(all_true_labels, all_predicted_labels)

    # Save confusion matrix and overall accuracy to the text file
    results_file.write("Confusion Matrix:\n")
    results_file.write(str(conf_matrix))
    results_file.write("\n\n")
    results_file.write(f"Overall Accuracy: {accuracy:.2%}\n")

    # Close the results file
    results_file.close()
