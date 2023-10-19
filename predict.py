import torch 
import modules as m
import dataset as d
from torch.utils.data import DataLoader
import random as r

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Using cpu.")

batch = 32

img_dir = "/home/groups/comp3710/ISIC2018/ISIC2018_Task1-2_Training_Input_x2"
seg_dir = "/home/groups/comp3710/ISIC2018/ISIC2018_Task1_Training_GroundTruth_x2"
test_dataset = d.ISICDataset(img_dir, seg_dir, d.transform('train'), d.transform('seg'))
test_loader = DataLoader(test_dataset, batch, shuffle=False)
model = m.ModifiedUNet(3, 1).to(device)

model.load_state_dict(torch.load('model_weights.pth'))



# if i == 1:
            #     test_image = images.cpu()[0].squeeze(0).numpy()
            #     # test_segment = segment.cpu()[0]
            #     # test_modelled_image = modelled_image.cpu()
            #     plt.plot(test_image)
            #     plt.title("test")
            #     # plt.imshow(test_segment.numpy())
            #     # plt.imshow(test_modelled_image.numpy())
            #     plt.savefig("/home/Student/s4742286/PatternAnalysis-2023/outputs/test.jpg")
            #     exit()
