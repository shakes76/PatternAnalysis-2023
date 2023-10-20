import argparse
import os
import matplotlib.pyplot as plt
import torch

from modules import ViT
from dataset import get_user_image

# Define command-line arguments
parser = argparse.ArgumentParser(description="Predict Alzheimer's Disease from Images")
parser.add_argument("--model_path", required=True, help="Path to model weights file")
parser.add_argument(
    "--image_path",
    required=True,
    default=None,
    help="Path to the image for prediction.",
)
parser.add_argument(
    "--output_folder",
    default=".",
    help="Path to the folder where prediction results will be saved",
)
args = parser.parse_args()

# Initialize the device
device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

if __name__ == "__main__":
    # Check if the specified image file exists
    if not os.path.exists(args.image_path):
        raise FileNotFoundError(f"Image file '{args.image_path}' not found.")

    # Load and preprocess the image
    print(f"\nFETCHING USER IMAGE\n{'='*25}\n")
    user_image = get_user_image(args.image_path).to(device)

    # Check if the specified image file exists
    if not os.path.exists(args.output_folder):
        raise FileNotFoundError(f"Output Folder '{args.output_folder}' not found.")

    # Define the base output folder
    output_folder = args.output_folder

    # Check if the specified model weights file exists
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model weights file '{args.model_path}' not found.")

    # Initialize Model
    print(f"\nINITIALIZING MODEL\n{'='*25}\n")
    print("Assigning model instance...")
    model = ViT(
        in_channels=1,
        patch_size=14,
        emb_size=1536,
        img_size=224,
        depth=10,
        n_classes=2,
    ).to(device)

    # Load trained model weights
    print("Loading Model Weights...")
    model.load_state_dict(torch.load(args.model_path))
    model.eval()
    print("Model ready.")

    # Predict class for the image
    print(f"\RUNNING PREDICTION\n{'='*25}\n")
    with torch.no_grad():
        logits = model(user_image)
    pred = logits.argmax(dim=1).item()  # Get the prediction as an integer

    # Define class labels
    class_labels = ["AD", "NC"]
    print(f"Image predicted to be of class: {class_labels[pred]}")

    print(f"\GENERATING VISUALIZATION\n{'='*25}\n")
    # Allocating user image tensor to cpu for visualization (can't do on mps)
    user_image = user_image.squeeze().cpu()

    # Visualize image with its predicted label
    plt.imshow(user_image)
    plt.title(f"Predicted: {class_labels[pred]}")
    image_filename = os.path.basename(args.image_path)
    image_filepath = os.path.join(output_folder, f"predicted_{image_filename}")
    plt.savefig(image_filepath)
    plt.close()

    print(f"Image saved to {image_filepath}")
