import torch
from glob import glob
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms


class MyDataset(torch.utils.data.Dataset):

    def __init__(self, mode, limit=None):
        '''
            folder_name: folder save brain
            seg_folder_name: folder save seg
            limit: only load limit data
        '''
        assert mode in ['train', 'test', 'validate']

        main_folder = '../../../keras_png_slices_data/keras_png_slices_data'
        # folder_name
        folder_name = f'{main_folder}/keras_png_slices_{mode}'
        seg_folder_name = f'{main_folder}/keras_png_slices_seg_{mode}'
        self.raw_img = []
        self.seg_img = []
        self.brain_index = []
        # Get images
        raw_paths = sorted(glob(f'{folder_name}/*.png'))
        seg_paths = sorted(glob(f'{seg_folder_name}/*.png'))

        total = limit if limit else len(raw_paths)
        for img_path, seg_path in tqdm(zip(raw_paths, seg_paths), total=total):
            # check if the image is paired
            assert tuple(img_path.split(
                '_')[-3:]) == tuple(seg_path.split('_')[-3:])

            # Brain Index: brain's CT. z_index: CT's z-index
            brain_index, _, z_index = img_path.split('_')[-3:]
            brain_index = int(brain_index)
            z_index = int(z_index.split('.')[0])

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
            self.brain_index.append((brain_index, z_index))

            # Only load limit images if limit has been set
            if limit and len(self.raw_img) == limit:
                break

        # Sorted the images by brain_index
        tmp = sorted([(j, k, i)
                      for i, (j, k) in enumerate(self.brain_index)])
        # Get rank given index in self.brain_index
        self.inv_index_dict = {i: i2 for i2, (j, k, i) in enumerate(tmp)}
        # Get index in self.brain_index given rank
        self.index_dict = {v: k for k, v in self.inv_index_dict.items()}

        self.raw_transform = transforms.Compose([
            transforms.ToTensor(),
            # make range(0, 1) -> range(-1, 1)
            transforms.Normalize((0.5, ), (0.5, ))
        ])
        self.seg_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.raw_img)

    def __getitem__(self, rank_idx):
        idx = self.index_dict[rank_idx]
        raw_img_tensor = self.raw_transform(self.raw_img[idx])
        seg_img_tensor = self.seg_transform(self.seg_img[idx])
        brain_idx, z_idx = self.brain_index[idx]
        return raw_img_tensor, (seg_img_tensor*3).long(), brain_idx, z_idx


__DATASET__ = {}


def get_dataloader(mode='train', batch_size=8, limit=None):
    assert mode in ['train', 'test', 'validate', 'train_and_validate', 'all']
    # To some issue, we may call get_dataloader twice.
    if (mode, limit) in __DATASET__:
        print("Call multiple times get_dataloader on same dataset. Reuse dataset")
        dataset = __DATASET__[(mode, limit)]
    elif mode == 'train_and_validate':
        # Build concat dataset that contains train & validate
        train_dataset = MyDataset(mode='train', limit=limit)
        validate_dataset = MyDataset(mode='validate', limit=limit)
        dataset = torch.utils.data.ConcatDataset(
            [train_dataset, validate_dataset])
    elif mode == 'all':
        # Build concat dataset that contains train & validate & test
        train_dataset = MyDataset(mode='train', limit=limit)
        validate_dataset = MyDataset(mode='validate', limit=limit)
        test_dataset = MyDataset(mode='test', limit=limit)
        dataset = torch.utils.data.ConcatDataset(
            [train_dataset, validate_dataset, test_dataset])
    else:
        dataset = MyDataset(mode=mode, limit=limit)

    # For some debug issue, if called dataset multiple times,
    # It'll save the result and avoid reading twice.
    __DATASET__[(mode, limit)] = dataset

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == 'train' or mode == 'train_and_validate'))

    return dataloader
