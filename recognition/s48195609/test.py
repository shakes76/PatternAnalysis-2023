# test.py

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from datasplit import load_data  # Modify your data loading function import here
from modules import build_vision_transformer
from parameter import MODEL_SAVE_PATH, INPUT_SHAPE, IMAGE_SIZE, PATCH_SIZE, NUM_PATCHES, NUM_HEADS, PROJECTION_DIM, HIDDEN_UNITS, DROPOUT_RATE, NUM_LAYERS, MLP_HEAD_UNITS

def predict(model, test_data):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    y_true = []
    y_pred = []

    with torch.no_grad():
        for data in test_data:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            predicted = torch.round(torch.sigmoid(outputs))
            y_true.append(labels.cpu().numpy())
            y_pred.append(predicted.cpu().numpy())

    y_true = torch.cat(y_true).squeeze().numpy()
    y_pred = torch.cat(y_pred).squeeze().numpy()

    # Calculate confusion matrix
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

if __name__ == '__main__':
    # Modify your data loading function to accommodate the test data split
    test = load_data()  # This assumes that load_data properly splits the test set

    # Create and load the model
    model = build_vision_transformer(INPUT_SHAPE, IMAGE_SIZE, PATCH_SIZE, NUM_PATCHES, NUM_HEADS, PROJECTION_DIM, HIDDEN_UNITS, DROPOUT_RATE, NUM_LAYERS, MLP_HEAD_UNITS)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    
    # Call the predict function to evaluate and display the confusion matrix
    predict(model, test)
