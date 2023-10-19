"""
Script to generate results from a trained model 

@author: Rodger Xiang s4642506
"""
from modules import ViT
from train import CONFIG
from dataset import load_data
from pathlib import Path
import utils
import torch
import torch.nn as nn

MODEL_NAME = "feasible-sun-174"

def test_model(model: nn.Module, 
                data_loader: torch.utils.data.DataLoader,
                device:str):
    test_acc = 0
    model.eval()
    # testing loop
    with torch.inference_mode():
        for batch, (X, y) in enumerate(data_loader):
            X, y = X.to(device), y.float().to(device)
            # mixed precision
            with torch.cuda.amp.autocast(dtype=torch.float16, enabled=True,):
                y_pred_logits = model(X).squeeze()
            #model accuracy
            acc =  utils.accuracy(y_pred_logits, y)
            test_acc += acc
        # average accuracy over epoch
        test_acc = test_acc / len(data_loader)
    return test_acc

if __name__ == "__main__":    
    #setup random seeds
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    #device agnostic code
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #load model weights
    model = utils.load_model(MODEL_NAME, device, CONFIG)
    
    #get test set
    _, _, test_loader = load_data(CONFIG.batch_size, CONFIG.img_size)
    
    #make predictions
    acc = test_model(model, test_loader, device)
    
    print(f"model accuracy = {acc * 100:.2f}%")