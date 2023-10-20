import torch 
import modules as m
import dataset as d
from torch.utils.data import DataLoader
import random as r
import matplotlib.pyplot as plt
import numpy as np

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        print("Using cpu.")

    batch = 32

    img_dir = "/home/groups/comp3710/ISIC2018/ISIC2018_Task1-2_Training_Input_x2"
    seg_dir = "/home/groups/comp3710/ISIC2018/ISIC2018_Task1_Training_GroundTruth_x2"
    test_dataset = d.CustomISICDataset(img_dir, seg_dir, d.transform('train'), d.transform('seg'))
    test_loader = DataLoader(test_dataset, batch, shuffle=False) # Load the data

    model = m.ModifiedUNet(3, 1).to(device) 
    model.load_state_dict(torch.load('model_weights_predict.pth')) #

    figure, axis = plt.subplots(3, 3, figsize=(5, 5))
    axis[0][0].set_title("Original Image") # The titles that will appear above each column
    axis[0][1].set_title("Ground Truth")
    axis[0][2].set_title("Modelled Mask")
    dice_list = []
    for i, input in enumerate(test_loader):
        if i == round(r.random()*80):
            with torch.no_grad():
                for j in range(3):
                    n = r.randint(0,31) # Random sampling from the batch
                    image = input[0].cpu()[n].permute(1,2,0) # Same transforms as the one in train.py
                    ground_truth = input[1].cpu()[n][0].float()/255
                    modelled_image = model(input[0].to(device))[0].cpu()[n][0].float()

                    axis[j][0].imshow(image.numpy())
                    axis[j][0].xaxis.set_visible(False)
                    axis[j][0].yaxis.set_visible(False)
    
                    axis[j][1].imshow(ground_truth.numpy(), cmap="gray")
                    axis[j][1].xaxis.set_visible(False)
                    axis[j][1].yaxis.set_visible(False)
    
                    axis[j][2].imshow(modelled_image.numpy(), cmap="gray")
                    axis[j][2].xaxis.set_visible(False)
                    axis[j][2].yaxis.set_visible(False)
                    figure.suptitle("Predicted Masks")

                    dice_score = dice_coefficient_nobatch(ground_truth, modelled_image).item()
                    dice_list.append(dice_score)

            plt.savefig(f"/home/Student/s4742286/PatternAnalysis-2023/predictoutputs/GroupedResultsComparisonPrediction_No{i} ")

        

    print("Mean dice coefficient across random sample: " + str(np.mean(dice_list)))

def dice_coefficient_nobatch(x_true: torch.Tensor(), x_model: torch.Tensor(), smooth=1e-5) -> torch.Tensor():
	""" This is the dice coefficient function made for the dataset. The Dice coefficient is a measure of the overlap between 2 sets, we can easily use this with tensors to test how much the segmentation mask and the ground truth overlap.
		Parameters:
			x_true: The ground truth mask
			x_model: The mask that the model returns
			smooth: A tiny number that prevents a division by 0 error.
		
		Returns:
			dice: A tensor that contains the average loss.
	"""
	intersection = torch.sum(x_true * x_model)
	sum_of_squares_model = torch.sum(torch.square(x_model))
	sum_of_squares_true = torch.sum(torch.square(x_true))
	dice_score = 1 - (2 * intersection + smooth) / (sum_of_squares_model + sum_of_squares_true + smooth) # Add smooth on both sides so we reduce the effect of it
	dice_score = torch.mean(dice_score)
	return dice_score

if __name__ == "__main__":
     main()