import warnings
warnings.filterwarnings("ignore")
import os, shutil, argparse
import numpy as np

# set random seed
np.random.seed(0)
'''
    This file help us to split the dataset.
    It's going to be a training set, a validation set, a test set.
    We need to get all the image data into --data_path
    Example:
        dataset/train/dog/*.(jpg, png, bmp, ...)
        dataset/train/cat/*.(jpg, png, bmp, ...)
        dataset/train/person/*.(jpg, png, bmp, ...)
        and so on...
    
    program flow:
    1. generate label.txt.
    2. rename --data_path.
    3. split dataset.
'''

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, default=r'dataset/train', help='all train path')
    parser.add_argument('--test_path', type=str, default=r'dataset/test', help='all test path')
    parser.add_argument('--label_path', type=str, default=r'dataset/label.txt', help='label txt save path')
    parser.add_argument('--val_size', type=float, default=0.2, help='size of val set')
    opt = parser.parse_known_args()[0]
    return opt

if __name__ == '__main__':
    opt = parse_opt()
    with open(opt.label_path, 'w+', encoding='utf-8') as f:
        f.write('\n'.join(os.listdir(opt.train_path))) # label's name
    import os

    path = "./dataset/train"

    class_ = os.listdir(path)
    patients = {}
    for i in class_:
        imag_path = path + os.sep + i
        a = []
        for j in os.listdir(imag_path):
            id = j.split("_")[0]
            a.append(id)
        patients[f"{i}"] = list(set(a))
    num_ad = len(patients["AD"])
    num_nc = len(patients["NC"])
    print("ad" + " " + f"{num_ad}")
    print("nc" + " " + f"{num_nc}")

    train_str_len = len(str(len(os.listdir(opt.train_path))))
    test_str_len = len(str(len(os.listdir(opt.test_path))))

    for idx, i in enumerate(os.listdir(opt.train_path)):
        os.rename(r'{}/{}'.format(opt.train_path, i), r'{}/{}'.format(opt.train_path, str(idx).zfill(train_str_len))) #rewrite dataset name start with 0,00 or 000

    for idx, i in enumerate(os.listdir(opt.test_path)):
        os.rename(r'{}/{}'.format(opt.test_path, i), r'{}/{}'.format(opt.test_path, str(idx).zfill(
            test_str_len)))  # rewrite dataset name start with 0,00 or 000

    os.chdir(opt.train_path)

    for index,i in enumerate(os.listdir(os.getcwd())):
        print(i)
        base_path = os.path.join(os.getcwd(), i)

        base_arr = os.listdir(base_path)
        if index == 0:
            k = "AD"
        else:
            k="NC"

        id = patients[k]

        val_path = base_path.replace('train', 'val')
        if not os.path.exists(val_path):
            os.makedirs(val_path)
        val_need_copy = id[int(len(id) * (1 - opt.val_size)):
                                 int(len(id))]

        for j in base_arr:
            if j.split("_")[0] in val_need_copy:
                shutil.move(os.path.join(base_path, j), os.path.join(val_path, j))



## run
## python processing.py