import torch

def predict_example(model, data):
    # Load the trained model
    model.load_state_dict(torch.load('path/to/saved/model.pth'))
    model.eval()

    # Perform prediction on example data
    with torch.no_grad():
        output = model(data)

    # Post-processing and interpretation of the prediction
    return output