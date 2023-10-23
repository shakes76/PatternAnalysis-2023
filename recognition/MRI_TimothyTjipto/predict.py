'''Shows example usage of trained model. Print out any results and/ or provide visualisations where applicable'''
import matplotlib.pyplot as plt
import torch

# Plotting data
def show_plot(iteration,loss):
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