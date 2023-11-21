import torch, tqdm
import torchvision.transforms as transforms

def get_mean_and_std(dataset, opt):
    '''Compute the mean and std value of dataset.'''
    if opt.imagenet_meanstd:
        print('using ImageNet Mean and Std. Mean:[0.485, 0.456, 0.406] Std:[0.229, 0.224, 0.225].')
        return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    else:
        print('Calculate the mean and variance of the dataset...')
        mean = torch.zeros(opt.image_channel)
        std = torch.zeros(opt.image_channel)

        for inputs, targets in tqdm.tqdm(dataset):
            inputs = transforms.ToTensor()(inputs)
            mean += inputs.mean()
            std += inputs.std()

        mean.div_(len(dataset))
        std.div_(len(dataset))
        print('Calculate complete. Mean:[{:.3f}] Std:[{:.3f}].'.format(mean.item(), std.item()))
        return mean, std


def get_processing(dataset, opt):
    return transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(*get_mean_and_std(dataset, opt)),
         ])


def select_Augment(opt):
    if opt.Augment == 'RandAugment':
        return transforms.RandAugment()
    elif opt.Augment == 'AutoAugment':
        return transforms.AutoAugment()
    elif opt.Augment == 'TrivialAugmentWide':
        return transforms.TrivialAugmentWide()
    elif opt.Augment == 'AugMix':
        return transforms.AugMix()
    else:
        return None

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized Tensor image.
        """
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

def get_dataprocessing(dataset, opt, preprocess=None):
    if not preprocess:
        preprocess = get_processing(dataset, opt)
        torch.save(preprocess, r'{}/preprocess.transforms'.format(opt.save_path))

    if len(opt.custom_augment.transforms) == 0:
        augment = select_Augment(opt)
    else:
        augment = opt.custom_augment

    if augment is None:
        train_transform = transforms.Compose(
            [transforms.Grayscale(num_output_channels=1),
            #transforms.Resize((int(opt.image_size + opt.image_size * 0.1))),
             transforms.CenterCrop((opt.image_size, opt.image_size)),
             transforms.RandomHorizontalFlip(),
             transforms.RandomRotation(degrees=10),
             preprocess,
             AddGaussianNoise(0., 0.05),
             ])
    else:
        train_transform = transforms.Compose(
            [transforms.Resize((int(opt.image_size + opt.image_size * 0.1))),
             transforms.RandomCrop((opt.image_size, opt.image_size)),
             augment,
             preprocess,
             ])


    val_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.CenterCrop((opt.image_size, opt.image_size)),
        preprocess
    ])

    return train_transform, val_transform

def get_dataprocessing_teststage(train_opt, opt, preprocess):
    test_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.CenterCrop((train_opt.image_size, train_opt.image_size)),
        preprocess
    ])
    return test_transform