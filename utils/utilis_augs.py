import torch, tqdm
import torchvision.transforms as transforms

def get_mean_and_std(dataset, opt):
    '''Compute the mean and std value of dataset.'''
    if opt.imagenet_meanstd:
        print('using ImageNet Mean and Std. Mean:[0.485, 0.456, 0.406] Std:[0.229, 0.224, 0.225].')
        return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    else:
        print('Calculate the mean and variance of the dataset...')
        mean = torch.zeros(1)
        std = torch.zeros(1)

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
         transforms.Normalize(*get_mean_and_std(dataset, opt))])


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
             preprocess
             ])
    else:
        train_transform = transforms.Compose(
            [transforms.Resize((int(opt.image_size + opt.image_size * 0.1))),
             transforms.RandomCrop((opt.image_size, opt.image_size)),
             augment,
             preprocess
             ])

    if opt.test_tta:
        val_transform = transforms.Compose([
            transforms.Resize((int(opt.image_size + opt.image_size * 0.1))),
            transforms.TenCrop((opt.image_size, opt.image_size)),
            transforms.Lambda(lambda crops: torch.stack([preprocess(crop) for crop in crops]))
        ])
    else:
        val_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            #transforms.Resize((opt.image_size)),
            transforms.CenterCrop((opt.image_size, opt.image_size)),
            preprocess
        ])

    return train_transform, val_transform

def get_dataprocessing_teststage(train_opt, opt, preprocess):
    if opt.test_tta:
        test_transform = transforms.Compose([
            transforms.Resize((int(train_opt.image_size + train_opt.image_size * 0.1))),
            transforms.TenCrop((train_opt.image_size, train_opt.image_size)),
            transforms.Lambda(lambda crops: torch.stack([preprocess(crop) for crop in crops]))
        ])
    else:
        test_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            #transforms.Resize((train_opt.image_size)),
            transforms.CenterCrop((train_opt.image_size, train_opt.image_size)),
            preprocess
        ])
    return test_transform