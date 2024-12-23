import train
import modules
import dataset
import torch
from PIL import Image
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = '/home/Student/s4722435/miniconda3/envs/new_torch/unet_stuff/model.pth'

def predict(loader):
    '''
    Produce the segmented image of given dataset (test dataset) and
    show the Dice Similarity Coefficient.
    '''
    model = modules.UNet().to(DEVICE)  
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()
    dsc = 0.0
    model.eval()
    for idx, (raw, seg) in loader:
        imgs, labels = raw.to(DEVICE), seg.to(DEVICE)
        outputs = model(imgs)
        output = outputs.squeeze().cpu().numpy()
        output = (output * 255).astype(np.uint8)
        output_img = Image.fromarray(output)
        output_img.save('seg_picture_' + str(idx) + '.png')
        dsc = train.dice_similarity_coeff(outputs, labels)
    dsc = dsc // len(loader)
    print('Disc Similarity Coefficient', dsc)

def main():
    test_set = dataset.keras_dataloader(256)
    predict(test_set[0])

if __name__ == '__main__':
    main()