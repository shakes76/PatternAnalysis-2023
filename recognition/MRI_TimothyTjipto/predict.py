'''Shows example usage of trained model. Print out any results and/ or provide visualisations where applicable'''

# Importing necessary libraries and modules
import matplotlib.pyplot as plt
import torch
from torchvision import transforms


def show_plot(iteration,loss):
    """
    Plots the loss values against iterations and saves the resulting graph.

    Args:
        iteration (List[int]): A list of iteration numbers.
        loss (List[float]): A list of loss values corresponding to each iteration.
    """
    plt.clf()
    plt.plot(iteration,loss)
    plt.savefig("Iteration loss")
    plt.show()

def predict(model, input1, input2):
    """
    Make predictions using a trained PyTorch model.
    
    Args:
    - model: Trained PyTorch model.
    - input_data: PyTorch tensor containing the input data.
    
    Returns:
    - Predictions as a PyTorch tensor.
    """
    # Set the model to evaluation mode
    model.eval()
    
    # Disable gradient computation
    with torch.no_grad():
        predictions = model(input1, input2)
    
    return predictions

def classify_pair(score, threshold):
    """
    Classify pairs of samples based on a threshold.
    
    Args:
    - score: Score of Dissimilarity
    - threshold: Decision boundary for classification.
    
    Returns:
    - List of classifications (0 for dissimilar, 1 for similar).
    """
    
    classification = 1 if score < threshold else 0
    return classification


def visual_pred_dis(idx,x0,x1,x0label,x1label,euclidean_distance,predict_class):
    """
        Visualizes the predictions by plotting the input images and their predicted dissimilarity.

    Args:
        idx (int): _description_
        x0 (torch.Tensor): First input image tensor
        x1 (torch.Tensor): Second input image tensor
        x0label (int): Label of first image
        x1label (int): Label of second image
        euclidean_distance (float): Calculated euclidean distance between the embeddings of the two images.
        predict_class (int): Predicted class (0 for 'Different', 1 for 'Same').
    """
    Prediction = ['Different', 'Same']

    plt.clf()
    plt.subplot(2, 8, 1)
    plt.title(int(x0label))
    x0_pic = transforms.ToPILImage()(x0[0])
    plt.axis('off')
    plt.imshow(x0_pic, cmap='gray')

    plt.subplot(2,8,2)
    plt.title(f'Dissimilarity: {euclidean_distance.item():.2f}\nClass predicted: {Prediction[predict_class]} ')
    plt.axis('off')



    plt.subplot(2, 8, 3)
    plt.title(int(x1label))
    x1_pic = transforms.ToPILImage()(x1[0])
    plt.axis('off')
    plt.imshow(x1_pic, cmap='gray')

    plt.savefig(f'/home/Student/s4653241/MRI/Test_pic/test{idx}')