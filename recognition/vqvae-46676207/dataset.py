from utils import *


class DataType(Enum):
    """
        Represents types of datasets
    """
    TRAIN = 1   # Training set
    VALID = 2   # Validating set
    TEST = 3    # Testing set

class OASIS_MRI(Dataset):
    def __init__(self, root, type=DataType.TRAIN, transform = None, target_transform=None) -> None:
        super(OASIS_MRI, self).__init__()
        
        self.type = type  # training set / valid set / test set
        self.transform = transform
        self.target_transform = target_transform

        if self.type == DataType.TRAIN:     # get training data
            file_annotation = root + TRAIN_TXT
            self.input_folder = TRAIN_INPUT_PATH
            # self.target_folder = TRAIN_TARGET_PATH
        elif self.type == DataType.VALID:   # get validating data
            file_annotation = root + VALID_TXT
            self.input_folder = VALID_INPUT_PATH
            # self.target_folder = VALID_TARGET_PATH
        elif self.type == DataType.TEST:    # get testing data
            file_annotation = root + TEST_TXT
            self.input_folder = TEST_INPUT_PATH
            # self.target_folder = TEST_TARGET_PATH

        f = open(file_annotation, 'r') # open file in read only
        data_dict = f.readlines() # get all lines from file
        f.close() # close file

        self.inputs = []
        # self.target_filenames = []
        self.labels = []
        for line in data_dict:
            img_names = line.split() # slipt by ' ', [0]: input, [1]: target
            input = Image.open(self.input_folder + img_names[0])    # read input img
            # m, s = np.mean(input, axis=(0, 1)), np.std(input, axis=(0, 1))
            preprocess = transforms.Compose([
                transforms.ToTensor(),
                # transforms.Normalize(mean=m, std=s),
            ])
            input = preprocess(input) # to tensore
            self.inputs.append(input)
            self.labels.append(img_names[1])

    def __getitem__(self, index):
        # index will be handled by dataloader
        input = self.inputs[index]
        label = int(self.labels[index])
        return input, label
    
    def __len__(self):
        return len(self.inputs)

def load_data(data_dir='./data',
                batch_size=256, # good for 4070ti
                random_seed=42,
                valid_size=0.1,
                shuffle=True,
                test=False) -> DataLoader :
    """
    Return a Dataloader of OASIS_MRI
    """
    normalize = transforms.Normalize(
        mean=[0.5],
        std=[0.2],
    )

    # define transforms
    transform = transforms.Compose([
            transforms.ToTensor(),
            # normalize,    # not working for OASIS_MRI
    ])

    if test:    # get the testing data
        test_dataset = OASIS_MRI(
          root=DATA_PATH, type=DataType.TEST,
          transform=transform,
        )

        data_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=shuffle
        )

        return data_loader
    
    else:       # get the training data & validating data
        train_dataset = OASIS_MRI(  # get training set
            root=DATA_PATH, type=DataType.TRAIN,
            transform=transform,
        )

        valid_dataset = OASIS_MRI(  # get validating set
            root=DATA_PATH, type=DataType.VALID,
            transform=transform,
        )

        s_train = len(train_dataset)  # size of training set
        indices = list(range(s_train))
        split = int(np.floor(valid_size * s_train))

        if shuffle:
            # shuffle the data set
            np.random.seed(random_seed)
            np.random.shuffle(indices)

        train_idx, valid_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=2
        )
    
        valid_loader = DataLoader(
            valid_dataset, batch_size=batch_size, sampler=valid_sampler, num_workers=2
        )

        return (train_loader, valid_loader)

def show_img(datatype=DataType.TRAIN):
    """
    show images in the house-made OASIS_MRI dataset
    """
    dataset = OASIS_MRI(DATA_PATH,type=datatype,transform=transforms.ToTensor())    # get the dataset
    dataloader = DataLoader(dataset=dataset, batch_size=64, shuffle=True)           # get the dataloader

    for step ,(b_x,b_y) in enumerate(dataloader): # b_x: input, b_y: target
        if step < 3:    # show 3 batches of images
            imgs = torchvision.utils.make_grid(b_x)
            imgs = np.transpose(imgs,(1,2,0))
            plt.imshow(imgs)
            plt.show()

if __name__ == "__main__":
    show_img()

