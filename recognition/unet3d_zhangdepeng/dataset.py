import nibabel as nib
# from nibabel import  niftil
# from nibabel.viewers import OrthoSlicer3D

from torch.utils.data import Dataset,DataLoader
import os
import torch
# example_filename = '/Users/tongxueqing/Downloads/HipMRI_study_complete_release_v1/semantic_MRs_anon/Case_004_Week0_LFOV.nii.gz'
# example_label = '/Users/tongxueqing/Downloads/HipMRI_study_complete_release_v1/semantic_labels_anon/Case_004_Week0_SEMANTIC_LFOV.nii.gz'

# img = nib.load(example_filename)
# img_data = img.get_fdata()
# img_affine = img.affine
# label = nib.load(example_label)
# label_data = label.get_fdata()
# print(img) 

from monai.transforms import (
    Compose,
    LoadImaged,
    RandSpatialCropd,
    EnsureTyped,
    CastToTyped,
    NormalizeIntensityd,
    RandFlipd,
    Lambdad,
    Resized,
    AddChanneld,
    RandGaussianNoised,
    RandGridDistortiond,
    RepeatChanneld,
    Transposed,
    OneOf,
    EnsureChannelFirstd,
    RandLambdad,
    Spacingd,
    FgBgToIndicesd,
    CropForegroundd,
    RandCropByPosNegLabeld,
    ToDeviced,
    SpatialPadd,

)
##for train split : resize and randomfip
train_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        
        
        EnsureChannelFirstd(keys=["image", "label"]),
        
       
        Lambdad(keys="image", func=lambda x:(x - x.min()) / (x.max()-x.min())),

        RandFlipd(keys=("image", "label"), prob=0.5, spatial_axis=[0]),
        RandFlipd(keys=("image", "label"), prob=0.5, spatial_axis=[1]),
        RandFlipd(keys=("image", "label"), prob=0.5, spatial_axis=[2]),
        Resized(keys=["image", "label"],spatial_size=(256,256,128)),
        EnsureTyped(keys=("image", "label"), dtype=torch.float32),
    ]
)
##for test split : just load
val_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        # Spacingd(keys=["image", "label"], pixdim=cfg.spacing, mode=("bilinear", "nearest")),
        Lambdad(keys="image", func=lambda x: (x - x.min()) / (x.max()-x.min())),
       
        EnsureTyped(keys=("image", "label"), dtype=torch.float32),
        # ToDeviced(keys=["image", "label"], device="cuda:0"),
    ]
)

class MRIDataset_pelvis(Dataset):
    """
     Code for reading Prostate 3D data set 
    """
    def __init__(self,mode,dataset_path='./HipMRI_study_complete_release_v1'):
        """
        param mode: 'train','val','test'
        param datset_path: root dataset folder
        """
        self.mode=mode
        self.CLASSES = 6
        self.vol_dim = (256,256,128)
        self.train_transform = train_transforms
        self.test_transform = val_transforms

            
        if self.mode == 'train':
#             select_list=os.listdir(os.path.join(dataset_path,'semantic_MRs_anon'))[:split_id]
            with open('train_list.txt','r') as f:
                y=f.readlines()
                select_list = [_.strip() for _ in y]
#                 print(y)
#                 print(len(y))
#                 print(y[0].strip())
            self.img_list = [os.path.join(dataset_path,'semantic_MRs_anon',_) for _ in select_list]
            self.label_list = [os.path.join(dataset_path,'semantic_labels_anon',_.replace('_LFOV','_SEMANTIC_LFOV')) for _ in select_list]
            self.train_files = [{'image':image_name,'label':label_name}  for image_name,label_name in zip(self.img_list,self.label_list)]

        elif self.mode == 'test':
            with open('test_list.txt','r') as f:
                y=f.readlines()
                select_list = [_.strip() for _ in y]
#             select_list=os.listdir(os.path.join(dataset_path,'semantic_MRs_anon'))[split_id:]
            self.img_list = [os.path.join(dataset_path,'semantic_MRs_anon',_) for _ in select_list]
            self.label_list = [os.path.join(dataset_path,'semantic_labels_anon',_.replace('_LFOV','_SEMANTIC_LFOV')) for _ in select_list]
            self.test_files = [{'image':image_name,'label':label_name}  for image_name,label_name in zip(self.img_list,self.label_list)]

    def __len__(self):
        return len(self.label_list)
    
    def __getitem__(self,index):
        img_path = self.img_list[index]
        label_path = self.label_list[index]
        # img = nib.load(img_path)
        # img_data = img.get_fdata()
        # label = nib.load(label_path)
        # label_data = label.get_fdata()
        # print(img_data.shape,label_data.shape)

        if self.mode == 'train' :
            # augmented_t1, augmented_s = self.train_transform(img_data,label_data)  
            augmented_t1 = self.train_transform({'image':img_path,'label':label_path})
            return augmented_t1['image'],augmented_t1['label']
            
        if self.mode == 'test':
            # augmented_t1, augmented_s = self.test_transform(img_data,label_data)
            augmented_t1 = self.test_transform({'image':img_path,'label':label_path})
            return augmented_t1['image'],augmented_t1['label']
            

        


if __name__=='__main__':
    test_dataset = MRIDataset_pelvis(mode='test',dataset_path='/root/HipMRI_study_complete_release_v1',normalize=True,augmentation=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=2, shuffle=False)
    print(len(test_dataset))
    for batch_ndx, sample in enumerate(test_dataloader):
        print('test')
        print(sample[0].shape)
        print(sample[1].shape)
        break
    train_dataset = MRIDataset_pelvis(mode='train',dataset_path='/root/HipMRI_study_complete_release_v1',normalize=True,augmentation=True)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=2, shuffle=False)
    for batch_ndx, sample in enumerate(train_dataloader):
        print('train')
        print(sample[0].shape)
        print(sample[1].shape)
        break



