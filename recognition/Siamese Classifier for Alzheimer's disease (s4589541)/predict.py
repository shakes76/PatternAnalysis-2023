# example usage of trained model
from dataset import *
from modules import *
import torch

def main(chkpt_path, model: TripletNetwork, criterion: TripletLoss,
          train_loader: DataLoader,):
    model.load_state_dict(torch.load(chkpt_path)
    model.eval()
    for batch_no, (a_v, label, p_v, n_v) in enumerate(valid_loader):
        # move the data to the GPU

        # input triplet images into model
        a_out_v, p_out_v, n_out_v = model(a_v, p_v, n_v)
        # calculate the loss
        loss_v = criterion(a_out_v, p_out_v, n_out_v)
        # add the loss
        epoch_valid_loss += loss_v.item()

        print(f"Batch {batch_no + 1}, Loss: {loss_v.item()}")