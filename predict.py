import torch
import os
from torchvision import transforms
from torch.utils.data import DataLoader
from modules import ConvolutionalVisionTransformer, CViTConfig
from process import load_test_data,configuration
from dataset import ADNC_Dataset, get_image_paths_from_directory, extract_patient_id

def test_model(model_path, test_imagesAD_path, test_images_nc_path, batch_size):
    """
    Evaluate the performance of trained CViT model on a test dataset. The pretrained model and test dataset is loaded, and the model's
    classification accuracy on the test data is evaluated. 
    """
    # Load test data
    test_images_AD, test_images_NC = load_test_data(test_imagesAD_path, test_images_nc_path)

    # Create data loader for testing
    data_transforms = {
        'test': transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    test_dataset = ADNC_Dataset(test_images_AD, test_images_NC, transform=data_transforms['test'])
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    config_params_dict=configuration()
    config = CViTConfig(config_params_dict)
    model = ConvolutionalVisionTransformer(config)
    
    # Load the trained model weights
    model.load_state_dict(torch.load(model_path))
    model.eval()


    def evaluate_model(model, dataloader):
        """
        Evaluate performance of model on dataset. All batches of dataset are gathered into a dataloader, and predictions are made using model. The 
        accuracy of the model is calculated by comparing predictions with actual label
        """
        model.eval()
        correct_predictions = 0
        total_samples = 0

        for images, labels in dataloader:
            with torch.no_grad():
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                correct_predictions += (predicted == labels).sum().item()
                total_samples += labels.size(0)

        accuracy = correct_predictions / len(dataloader.dataset)
        return accuracy

    # Evaluate the model on the test set
    test_accuracy = evaluate_model(model, test_dataloader)
    print(f'Test Accuracy: {test_accuracy:.4f}')
