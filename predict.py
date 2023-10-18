import torch
from modules import ViT

def predict(workers: int):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ViT(workers).to(device)
    model.load_state_dict(torch.load('visual_transformer'))
    model.eval()


if __name__ == '__main__':
    predict()