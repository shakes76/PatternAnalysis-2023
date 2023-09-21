import torch
from glob import glob
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms

class MyDataset(torch.utils.data.Dataset):

    def __init__(self, folder_name, seg_folder_name):
        self.raw_img = []
        self.seg_img = []

        # Get images
        raw_paths = sorted(glob(f'{folder_name}/*.png'))
        seg_paths = sorted(glob(f'{seg_folder_name}/*.png'))
        for img_path, seg_path in tqdm(zip(raw_paths, seg_paths), total=len(raw_paths)):
            # check if the image is paired
            assert tuple(img_path.split('_')[-3:]) == tuple(seg_path.split('_')[-3:])


            raw_img = Image.open(img_path)
            seg_img = Image.open(seg_path)

            # Get File Descriptor
            image_fp = raw_img.fp
            raw_img.load()

            # Close File Descriptor (or it'll reach OPEN_MAX)
            image_fp.close()

            # Get File Descriptor
            image_fp = seg_img.fp
            seg_img.load()

            # Close File Descriptor (or it'll reach OPEN_MAX)
            image_fp.close()

            # seg_img label = 4
            # print(raw_img.size, seg_img.size, len(np.unique(raw_img)), len(np.unique(seg_img)))
            # All Image size should be (256, 256)
            assert raw_img.size == seg_img.size == (256, 256)
            self.raw_img.append(raw_img)
            self.seg_img.append(seg_img)

        self.transform = transforms.Compose([
            # transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])
    def __len__(self):
        return len(self.raw_img)

    def __getitem__(self, idx):
        raw_img_tensor = self.transform(self.raw_img[idx])
        seg_img_tensor = self.transform(self.seg_img[idx])
        return raw_img_tensor, (seg_img_tensor*3).long()
    
def get_dataloader(mode='train', batch_size=8):

    assert mode in ['train', 'test', 'validate']
    dataset = MyDataset(
        f'../keras_png_slices_data/keras_png_slices_data/keras_png_slices_{mode}',
        f'../keras_png_slices_data/keras_png_slices_data/keras_png_slices_seg_{mode}')

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == 'train'))

    return dataloader

