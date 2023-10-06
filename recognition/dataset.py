from mrcnn.utils import Dataset
import cv2
import os

class SkinLesionDataset(Dataset):
    def load_dataset(self, dataset_dir):
        self.add_class("dataset", 1, "skin_lesion")
        images_dir = os.path.join(dataset_dir, 'ISIC2018_Task1-2_Training_Input_x2')
        annotations_dir = os.path.join(dataset_dir, 'ISIC2018_Task1_Training_GroundTruth_x2')

        for filename in os.listdir(images_dir):
            image_id = filename[:-4]
            img_path = os.path.join(images_dir, filename)
            ann_path = os.path.join(annotations_dir, image_id + '.png')
            self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        path = info['annotation']
        mask = cv2.imread(path)
        return mask, [1]
