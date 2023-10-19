import unittest
from CONFIG import *
import torch
from torch.utils.data import DataLoader, random_split
import cv2
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms import v2
from dataset import ISICDataloader

class DatasetTest(unittest.TestCase):

    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.dataloader = ISICDataloader(classify_file=classify_file, 
                                                 photo_dir=photo_dir, 
                                                 mask_dir=mask_dir,
                                                 mask_empty_dim=(1000, 700))


    def test_mask_to_bbox(self):
        mask = torch.Tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 2, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0]
                ])
        expected_bbox = [2, 1, 7, 5]
        calculated_bbox = self.dataloader.mask_to_bbox(mask=mask)
        self.assertListEqual(expected_bbox, calculated_bbox)

class ManualTest():
    def __init__(self):
        self.dataloader = ISICDataloader(classify_file=classify_file, 
                                                 photo_dir=photo_dir, 
                                                 mask_dir=mask_dir,
                                                 mask_empty_dim=(1000, 700))
        self.main()
        
    def main(self):
        self.test_dataloader()
        self.getNormalize()
    
    def test_dataloader(self):
        train_dataset, test_dataset = random_split(self.dataloader, [train_size, test_size])

        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        count = 0
        for raw_image, raw_target in test_dataset:
            image = raw_image.cpu().numpy().transpose((1, 2, 0))
            #image = np.transpose(raw_image, (2, 0, 1))
            ax = axes[count % 2, count // 2]
            ax.imshow(image)
            if len(raw_target['labels']) == 1:
                ax.set_title(raw_target['labels'][0].item())
                
                bbox = raw_target['boxes'].cpu().numpy()
                x1, y1, x2, y2 = bbox[0]
                ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color='red', linewidth=2))

                mask = raw_target['masks'].cpu().numpy()[0]
                masked_image = np.copy(image)
                masked_image[mask > 0.5] = [0, 255, 0]
                ax.imshow(masked_image, alpha=0.3)
            else:
                ax.set_title('none')
            ax.axis('off')

            count += 1
            if count > 6 - 1:
                 break
        plt.show()

    def getNormalize(self):
        transforms = v2.Compose([
            v2.ToImage(),
            v2.Resize(size=IMAGE_SIZE, antialias=True)
        ])
        
        mean = [0.0, 0.0, 0.0]
        std = [0.0, 0.0, 0.0]

        dataset = ISICDataloader(classify_file=classify_file, 
                                                 photo_dir=photo_dir, 
                                                 mask_dir=mask_dir,
                                                 mask_empty_dim=IMAGE_SIZE,
                                                 transform=transforms)

        for image, _ in dataset:
            for i in range(3):  # 3 channels: Red, Green, Blue
                mean[i] += image[i, :, :].mean()
                std[i] += image[i, :, :].std()
        
        num_samples = len(dataset)
        mean = [m / num_samples for m in mean]
        std = [s / num_samples for s in std]
        
        print(mean)
        print(std)




def main():
    ##unittest.main()
    ManualTest()
    


if __name__ == "__main__":
    main()