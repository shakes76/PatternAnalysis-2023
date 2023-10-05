import torch
from pathlib import Path

def accuracy(y_pred, y):
    y_pred_label = torch.round(torch.sigmoid(y_pred))
    correct = torch.eq(y, y_pred_label).sum().item()
    acc = (correct / len(y))
    return acc

def save_model(model, model_name):
    MODEL_PATH = Path("models")
    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    torch.save(obj=model.state_dict(), f=MODEL_PATH / model_name)

def load_model(path, model, device):
    model.load_state_dict(torch.load(f=path))
    return model.to(device)
