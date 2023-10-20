import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import modules as m
import dataset as d
import time
import matplotlib.pyplot as plt

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        print("Using cpu.")

    test_set_dice_list = []

    # These are the hyper parameters for the training.
    epochs = 30
    learning_rate = 0.0001
    batch = 32

    # Initialise the model
    model = m.ModifiedUNet(3, 1).to(device)

    # Directories for the image files given
    img_dir = "/home/groups/comp3710/ISIC2018/ISIC2018_Task1-2_Training_Input_x2"
    seg_dir = "/home/groups/comp3710/ISIC2018/ISIC2018_Task1_Training_GroundTruth_x2"

    # Preparing the data
    train_dataset = d.CustomISICDataset(img_dir, seg_dir, d.transform('train'), d.transform('seg'))
    train_loader = DataLoader(train_dataset, batch, shuffle=True)

    # We will use the ADAM optimizer
    ADAMoptimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # Now we begin timing
    starttime = time.time()
    for epoch in range(epochs):
        losslist = []
        runningloss_val = []
        runningloss = 0.0

        # Begin the training phase. 
        # In this phase, the model will receive images and put them through all the layers to approximate the mask for the image. 
        # We use the dice loss function to measure how well it's doing, or how much lossiness there is.
        model.train()

        for i, input in enumerate(train_loader):
            if i <= 55:
                images, segments = input[0].to(device), input[1].to(device) # Isolating the image and segment

                # Changes the grads from 0s to None's
                ADAMoptimizer.zero_grad()

                # Modelling the image
                modelled_images = model(images)[0]

                loss = m.dice_loss(modelled_images, segments)
                loss.backward() 
                ADAMoptimizer.step()
                runningloss += loss.item()
                losslist.append(loss.item())
                if i % 10 == 0:
                    print(f"Training: Epoch {epoch + 1}/{epochs}, Images {i }/55")
            elif i in range(56, 81):
                # Disables gradient calculations for increased performance
                with torch.no_grad():
                    # Validation phase is where we test the model to test if the model is doing well or not. The key difference here is that the optimizer plays no role in the calculations, as we presume we have found the minima that results in the least loss.
                    model.eval()
                    images, segments = input[0].to(device), input[1].to(device)

                    modelled_images = model(images)[0]
                    loss = m.dice_loss(modelled_images, segments)
                    runningloss_val.append(loss.item())
                    dice_score = 1 - m.dice_loss(modelled_images, segments) # The dice score is the complement of the dice loss function, so +1
                    test_set_dice_list.append(dice_score.item())
                if i % 10 == 0:
                    print(f"Validating: Epoch {epoch + 1}/{epochs}, Images {i - 56}/25")
            else:
                if epoch in [0, 9, 19, 29]:
                # This is the comparison of different masks made for different inputs at the end of select epochs to showcase change.
                    figure, axis = plt.subplots(2, 3, figsize=(5, 5))
                    axis[0][0].set_title("Original Image") # The titles that will appear above each column
                    axis[0][1].set_title("Ground Truth")
                    axis[0][2].set_title("Modelled Mask")

                    for row in range(2):
                        with torch.no_grad():
                            # Putting the tensors in the formatting necessary for matplotlib.pyplot
                            image = input[0].cpu()[row].permute(1,2,0)
                            ground_truth = input[1].cpu()[row][0].float()/255 
                            modelled_image = model(input[0].to(device))[0].cpu()[row][0].float()

                            axis[row][0].imshow(image.numpy())
                            axis[row][0].xaxis.set_visible(False)
                            axis[row][0].yaxis.set_visible(False)

                            axis[row][1].imshow(ground_truth.numpy(), cmap="gray")
                            axis[row][1].xaxis.set_visible(False)
                            axis[row][1].yaxis.set_visible(False)

                            axis[row][2].imshow(modelled_image.numpy(), cmap="gray")
                            axis[row][2].xaxis.set_visible(False)
                            axis[row][2].yaxis.set_visible(False)

                            figure.suptitle(f"Validation Phase: Epoch {epoch + 1}")
                    
                    # Saving the figure
                    plt.savefig(f"/home/Student/s4742286/PatternAnalysis-2023/outputs/GroupedResultsComparison_Epoch{epoch + 1} ")
                    plt.clf()


        # Easier to imagine epochs if they are 1-indexed instead of 0-indexed
        if epoch in [4, 9, 14, 19, 24, 29]:
            plt.plot(losslist)
            plt.xlabel('Epoch')
            plt.ylabel('Dice Loss')
            plt.title(f'Training Loss to Epoch {epoch + 1}')
            plt.savefig(f"/home/Student/s4742286/PatternAnalysis-2023/outputs/Training_Loss_Epoch_{epoch + 1}.png")
            plt.clf()

            plt.plot(runningloss_val)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(f'Validation Loss to Epoch {epoch + 1}')
            plt.savefig(f"/home/Student/s4742286/PatternAnalysis-2023/outputs/Validation_Loss_Epoch_{epoch + 1}.png")
            plt.clf()

    # Calculate the overall dice score
    test_dice_score = np.mean(test_set_dice_list)

    print(f"Testing finished. Time taken was {str(time.time()/60 - starttime/60)} minutes. Overall, the dice score that the model was able to provide was {test_dice_score}")

    # Save the model with the weights.
    torch.save(model.state_dict(), "model_weights.pth")

if __name__ == "__main__":
    main()