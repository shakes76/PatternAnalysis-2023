import pandas as pd
import yaml, os, torch, platform, itertools
from matplotlib import pyplot as plt
from prettytable import PrettyTable
import torch.nn as nn
from collections import OrderedDict
import numpy as np
from pycm import ConfusionMatrix
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report, confusion_matrix
from math import cos, pi
from PIL import Image

cnames = {
'aliceblue':   '#F0F8FF',
'antiquewhite':   '#FAEBD7',
'aqua':     '#00FFFF',
'aquamarine':   '#7FFFD4',
'azure':    '#F0FFFF',
'beige':    '#F5F5DC',
'bisque':    '#FFE4C4',
'black':    '#000000',
'blanchedalmond':  '#FFEBCD',
'blue':     '#0000FF',
'blueviolet':   '#8A2BE2',
'brown':    '#A52A2A',
'burlywood':   '#DEB887',
'cadetblue':   '#5F9EA0',
'chartreuse':   '#7FFF00',
'chocolate':   '#D2691E',
'coral':    '#FF7F50',
'cornflowerblue':  '#6495ED',
'cornsilk':    '#FFF8DC',
'crimson':    '#DC143C',
'cyan':     '#00FFFF',
'darkblue':    '#00008B',
'darkcyan':    '#008B8B',
'darkgoldenrod':  '#B8860B',
'darkgray':    '#A9A9A9',
'darkgreen':   '#006400',
'darkkhaki':   '#BDB76B',
'darkmagenta':   '#8B008B',
'darkolivegreen':  '#556B2F',
'darkorange':   '#FF8C00',
'darkorchid':   '#9932CC',
'darkred':    '#8B0000',
'darksalmon':   '#E9967A',
'darkseagreen':   '#8FBC8F',
'darkslateblue':  '#483D8B',
'darkslategray':  '#2F4F4F',
'darkturquoise':  '#00CED1',
'darkviolet':   '#9400D3',
'deeppink':    '#FF1493',
'deepskyblue':   '#00BFFF',
'dimgray':    '#696969',
'dodgerblue':   '#1E90FF',
'firebrick':   '#B22222',
'floralwhite':   '#FFFAF0',
'forestgreen':   '#228B22',
'fuchsia':    '#FF00FF',
'gainsboro':   '#DCDCDC',
'ghostwhite':   '#F8F8FF',
'gold':     '#FFD700',
'goldenrod':   '#DAA520',
'gray':     '#808080',
'green':    '#008000',
'greenyellow':   '#ADFF2F',
'honeydew':    '#F0FFF0',
'hotpink':    '#FF69B4',
'indianred':   '#CD5C5C',
'indigo':    '#4B0082',
'ivory':    '#FFFFF0',
'khaki':    '#F0E68C',
'lavender':    '#E6E6FA',
'lavenderblush':  '#FFF0F5',
'lawngreen':   '#7CFC00',
'lemonchiffon':   '#FFFACD',
'lightblue':   '#ADD8E6',
'lightcoral':   '#F08080',
'lightcyan':   '#E0FFFF',
'lightgoldenrodyellow': '#FAFAD2',
'lightgreen':   '#90EE90',
'lightgray':   '#D3D3D3',
'lightpink':   '#FFB6C1',
'lightsalmon':   '#FFA07A',
'lightseagreen':  '#20B2AA',
'lightskyblue':   '#87CEFA',
'lightslategray':  '#778899',
'lightsteelblue':  '#B0C4DE',
'lightyellow':   '#FFFFE0',
'lime':     '#00FF00',
'limegreen':   '#32CD32',
'linen':    '#FAF0E6',
'magenta':    '#FF00FF',
'maroon':    '#800000',
'mediumaquamarine':  '#66CDAA',
'mediumblue':   '#0000CD',
'mediumorchid':   '#BA55D3',
'mediumpurple':   '#9370DB',
'mediumseagreen':  '#3CB371',
'mediumslateblue':  '#7B68EE',
'mediumspringgreen': '#00FA9A',
'mediumturquoise':  '#48D1CC',
'mediumvioletred':  '#C71585',
'midnightblue':   '#191970',
'mintcream':   '#F5FFFA',
'mistyrose':   '#FFE4E1',
'moccasin':    '#FFE4B5',
'navajowhite':   '#FFDEAD',
'navy':     '#000080',
'oldlace':    '#FDF5E6',
'olive':    '#808000',
'olivedrab':   '#6B8E23',
'orange':    '#FFA500',
'orangered':   '#FF4500',
'orchid':    '#DA70D6',
'palegoldenrod':  '#EEE8AA',
'palegreen':   '#98FB98',
'paleturquoise':  '#AFEEEE',
'palevioletred':  '#DB7093',
'papayawhip':   '#FFEFD5',
'peachpuff':   '#FFDAB9',
'peru':     '#CD853F',
'pink':     '#FFC0CB',
'plum':     '#DDA0DD',
'powderblue':   '#B0E0E6',
'purple':    '#800080',
'red':     '#FF0000',
'rosybrown':   '#BC8F8F',
'royalblue':   '#4169E1',
'saddlebrown':   '#8B4513',
'salmon':    '#FA8072',
'sandybrown':   '#FAA460',
'seagreen':    '#2E8B57',
'seashell':    '#FFF5EE',
'sienna':    '#A0522D',
'silver':    '#C0C0C0',
'skyblue':    '#87CEEB',
'slateblue':   '#6A5ACD',
'slategray':   '#708090',
'snow':     '#FFFAFA',
'springgreen':   '#00FF7F',
'steelblue':   '#4682B4',
'tan':     '#D2B48C',
'teal':     '#008080',
'thistle':    '#D8BFD8',
'tomato':    '#FF6347',
'turquoise':   '#40E0D0',
'violet':    '#EE82EE',
'wheat':    '#F5DEB3',
'white':    '#FFFFFF',
'whitesmoke':   '#F5F5F5',
'yellow':    '#FFFF00',
'yellowgreen':   '#9ACD32'}

def save_model(path, **ckpt):
    torch.save(ckpt, path)

def str2float(data):
    return (0.0 if type(data) is str else data)

def setting_optimizer(opt, model):
    g = [], [], []  # optimizer parameter groups
    bn = tuple(v for k, v in nn.__dict__.items() if 'Norm' in k)  # normalization layers, i.e. BatchNorm2d()
    for v in model.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias (no decay)
            g[2].append(v.bias)
        if isinstance(v, bn):  # weight (no decay)
            g[1].append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # weight (with decay)
            g[0].append(v.weight)

    name = opt.optimizer
    if name == 'AdamW':
        optimizer = torch.optim.AdamW(g[2], lr=opt.lr, betas=(opt.momentum, 0.999), weight_decay=0.0)
    elif name == 'RMSProp':
        optimizer = torch.optim.RMSprop(g[2], lr=opt.lr, momentum=opt.momentum)
    elif name == 'SGD':
        optimizer = torch.optim.SGD(g[2], lr=opt.lr, momentum=opt.momentum, nesterov=True)

    optimizer.add_param_group({'params': g[0], 'weight_decay': opt.weight_decay})  # add g0 with weight_decay
    optimizer.add_param_group({'params': g[1], 'weight_decay': 0.0})  # add g1 (BatchNorm2d weights)

    return optimizer

def show_config(opt):
    table = PrettyTable()
    table.title = 'Configurations'
    table.field_names = ['params', 'values']
    opt = vars(opt)
    for key in opt:
        table.add_row([str(key), str(opt[key]).replace('\n', '')])
    print(table)

    if not opt['resume']:
        for keys in opt.keys():
            if type(opt[keys]) is not str:
                opt[keys] = str(opt[keys]).replace('\n', '')
            else:
                opt[keys] = opt[keys].replace('\n', '')

        with open(os.path.join(opt['save_path'], 'param.yaml'), 'w+') as f:
            # f.write(json.dumps(opt, indent=4, separators={':', ','}))
            yaml.dump(opt, f)

def dict_to_PrettyTable(data, name):
    data_keys = list(data.keys())
    table = PrettyTable()
    table.title = name
    table.field_names = data_keys
    table.add_row(['{:.5f}'.format(data[i]) for i in data_keys])
    return str(table)


def update_opt(a, b):
    #print(b)
    b = vars(b)
    for key in b:
        setattr(a, str(key), b[key])
    return a

def select_device(device='', batch_size=0):
    device = str(device).strip().lower().replace('cuda:', '').replace('none', '')  # to string, 'cuda:0' to '0'
    cpu = device == 'cpu'
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    elif device:
        os.environ['CUDA_VISIBLE_DEVICES'] = device
        assert torch.cuda.is_available() and torch.cuda.device_count() >= len(device.replace(',', '')), \
            f"Invalid CUDA '--device {device}' requested, use '--device cpu' or pass valid CUDA device(s)"

    print_str = f'Image-Classifier Python-{platform.python_version()} Torch-{torch.__version__} '
    if not cpu and torch.cuda.is_available():
        devices = device.split(',') if device else '0'
        n = len(devices)  # device count
        if n > 1 and batch_size > 0:  # check batch_size is divisible by device_count
            assert batch_size % n == 0, f'batch-size {batch_size} not multiple of GPU count {n}'
        space = ' ' * len(print_str)
        for i, d in enumerate(devices):
            p = torch.cuda.get_device_properties(i)
            print_str += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / (1 << 20):.0f}MiB)\n"
        arg = 'cuda:0'
    else:
        print_str += 'CPU'
        arg = 'cpu'
    print(print_str)
    return torch.device(arg)


def load_weights(model, opt):
    if opt.weight:
        if not os.path.exists(opt.weight):
            print('opt.weight not found, skipping...')
        else:
            print('found weight in {}, loading...'.format(opt.weight))
            state_dict = torch.load(opt.weight)
            if type(state_dict) is dict:
                try:
                    state_dict = state_dict['model'].state_dict()
                except:
                    pass
            elif not (state_dict is OrderedDict):
                state_dict = state_dict.state_dict()
            model = load_weights_from_state_dict(model, state_dict)
    return model

def load_weights_from_state_dict(model, state_dict):
    model_dict = model.state_dict()
    weight_dict = {}
    for k, v in state_dict.items():
        if k in model_dict:
            if np.shape(model_dict[k]) == np.shape(v):
                weight_dict[k] = v
    unload_keys = list(set(model_dict.keys()).difference(set(weight_dict.keys())))
    if len(unload_keys) == 0:
        print('all keys is loading.')
    elif len(unload_keys) <= 50:
        print('unload_keys:{} unload_keys_len:{} unload_keys/weight_keys:{:.3f}%'.format(','.join(unload_keys), len(unload_keys), len(unload_keys) / len(model_dict) * 100))
    else:
        print('unload_keys:{}.... unload_keys_len:{} unload_keys/weight_keys:{:.3f}%'.format(','.join(unload_keys[:50]), len(unload_keys), len(unload_keys) / len(model_dict) * 100))
    pretrained_dict = weight_dict
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model


class WarmUpLR:
    def __init__(self, optimizer, opt):
        self.optimizer = optimizer
        self.lr_min = opt.warmup_minlr
        self.lr_max = opt.lr
        self.max_epoch = opt.epoch
        self.current_epoch = 0
        self.lr_scheduler = opt.lr_scheduler(optimizer,
                                             **opt.lr_scheduler_params) if opt.lr_scheduler is not None else None
        self.warmup_epoch = int(opt.warmup_ratios * self.max_epoch) if opt.warmup else 0
        if opt.warmup:
            self.step()

    def step(self):
        self.adjust_lr()
        self.current_epoch += 1

    def adjust_lr(self):
        if self.current_epoch <= self.warmup_epoch and self.warmup_epoch != 0:
            lr = (self.lr_max - self.lr_min) * (self.current_epoch / self.warmup_epoch) + self.lr_min
        else:
            if self.lr_scheduler:
                self.lr_scheduler.step()
                return
            else:
                lr = self.lr_min + (self.lr_max - self.lr_min) * (
                        1 + cos(
                    pi * (self.current_epoch - self.warmup_epoch) / (self.max_epoch - self.warmup_epoch))) / 2
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def state_dict(self):
        return {
            'lr_min': self.lr_min,
            'lr_max': self.lr_max,
            'max_epoch': self.max_epoch,
            'current_epoch': self.current_epoch,
            'warmup_epoch': self.warmup_epoch,
            'lr_scheduler': self.lr_scheduler.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }

    def load_state_dict(self, state_dict):
        self.lr_min = state_dict['lr_min']
        self.lr_max = state_dict['lr_max']
        self.max_epoch = state_dict['max_epoch']
        self.current_epoch = state_dict['current_epoch']
        self.warmup_epoch = state_dict['warmup_epoch']
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.lr_scheduler.load_state_dict(state_dict['lr_scheduler'])


class Train_Metrice:
    def __init__(self, class_num):
        self.train_cm = np.zeros((class_num, class_num))
        self.test_cm = np.zeros((class_num, class_num))
        self.class_num = class_num

        self.train_loss = []
        self.test_loss = []

    def update_y(self, y_true, y_pred, isTest=False):
        if isTest:
            self.test_cm += cal_cm(y_true, y_pred, self.class_num)
        else:
            self.train_cm += cal_cm(y_true, y_pred, self.class_num)

    def update_loss(self, loss, isTest=False):
        if isTest:
            self.test_loss.append(loss)
        else:
            self.train_loss.append(loss)

    def get(self):
        result = {}
        result['train_loss'] = np.mean(self.train_loss)
        result['val_loss'] = np.mean(self.test_loss)
        result['train_acc'] = np.diag(self.train_cm).sum() / (self.train_cm.sum() + 1e-7)
        result['val_acc'] = np.diag(self.test_cm).sum() / (self.test_cm.sum() + 1e-7)
        result['train_mean_acc'] = np.diag(
            self.train_cm.astype('float') / self.train_cm.sum(axis=1)[:, np.newaxis]).mean()
        result['val_mean_acc'] = np.diag(self.test_cm.astype('float') / self.test_cm.sum(axis=1)[:, np.newaxis]).mean()
        cols_name = ['train_loss', 'train_acc', 'train_mean_acc', 'val_loss', 'val_acc', 'val_mean_acc']


        return result, ','.join(['{:.6f}'.format(result[i]) for i in cols_name])

def cal_cm(y_true, y_pred, CLASS_NUM):
    y_true, y_pred = y_true.to('cpu').detach().numpy(), np.argmax(y_pred.to('cpu').detach().numpy(), axis=1)
    y_true, y_pred = y_true.reshape((-1)), y_pred.reshape((-1))
    cm = confusion_matrix(y_true, y_pred, labels=list(range(CLASS_NUM)))
    return cm


class Model_Inference:
    def __init__(self, device, opt):
        self.opt = opt
        self.device = device

        if self.opt.model_type == 'torch':
            ckpt = torch.load(os.path.join(opt.save_path, 'best.pt'))
            self.model = ckpt['model'].float()
            model_fuse(self.model)
            self.model = (self.model.half() if opt.half else self.model)
            self.model.to(self.device)
            self.model.eval()


    def __call__(self, inputs):
        if self.opt.model_type == 'torch':
            with torch.inference_mode():
                return self.model(inputs)


    def forward_features(self, inputs):
        try:
            return self.model.forward_features(inputs)
        except:
            raise 'this model is not a torch model.'

    def cam_layer(self):
        try:
            return self.model.cam_layer()
        except:
            raise 'this model is not a torch model.'



def model_fuse(model):
    before_fuse_layers = len(getLayers(model))
    for module in model.modules():
        if hasattr(module, 'switch_to_deploy'):
            module.switch_to_deploy()
    print(f'model fuse... {before_fuse_layers} layers to {len(getLayers(model))} layers')


def fuse_conv_bn(conv, bn):
    # Fuse convolution and batchnorm layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    fusedconv = (
        nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            groups=conv.groups,
            bias=True,
        )
        .requires_grad_(False)
        .to(conv.weight.device)
    )

    # prepare filters
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    # prepare spatial bias
    b_conv = (
        torch.zeros(conv.weight.size(0), device=conv.weight.device)
        if conv.bias is None
        else conv.bias
    )
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(
        torch.sqrt(bn.running_var + bn.eps)
    )
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)
    return fusedconv

def getLayers(model):
    """
    get each layer's name and its module
    :param model:
    :return: each layer's name and its module
    """
    layers = []

    def unfoldLayer(model):
        """
        unfold each layer
        :param model: the given model or a single layer
        :param root: root name
        :return:
        """

        # get all layers of the model
        layer_list = list(model.named_children())
        for item in layer_list:
            module = item[1]
            sublayer = list(module.named_children())
            sublayer_num = len(sublayer)

            # if current layer contains sublayers, add current layer name on its sublayers
            if sublayer_num == 0:
                layers.append(module)
            # if current layer contains sublayers, unfold them
            elif isinstance(module, torch.nn.Module):
                unfoldLayer(module)

    unfoldLayer(model)
    return layers


class Metrice_Dataset(torch.utils.data.dataset.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        pil_img = Image.open(self.dataset.imgs[index][0])
        if pil_img.mode != 'L':
            pil_img = pil_img.convert('L')
        pil_img = self.dataset.transform(pil_img)
        return pil_img, self.dataset.imgs[index][1], self.dataset.imgs[index][0]


def classification_metrice(y_true, y_pred, class_num, label, save_path):
    metrice = Test_Metrice(y_true, y_pred, class_num)
    class_report, cm = metrice()
    class_pa = np.diag(cm.to_array(normalized=True))  # mean class accuracy
    if class_num <= 50:
        plot_confusion_matrix(cm.to_array(), label, save_path)
    save_confusion_matrix(cm.to_array(normalized=True), label, save_path)

    table1_cols_name = ['class'] + metrice.metrice_name
    table1 = PrettyTable()
    table1.title = 'Per Class'
    table1.field_names = table1_cols_name
    with open(os.path.join(save_path, 'perclass_result.csv'), 'w+', encoding='utf-8') as f:
        f.write(','.join(table1_cols_name) + '\n')
        for i in range(class_num):
            table1.add_row([label[i]] + ['{:.5f}'.format(class_report[i][j]) for j in table1_cols_name[1:-1]] + [
                '{:.5f}'.format(class_pa[i])])
            f.write(','.join([label[i]] + ['{:.5f}'.format(class_report[i][j]) for j in table1_cols_name[1:-1]] + [
                '{:.5f}'.format(class_pa[i])]) + '\n')
    print(table1)

    table2_cols_name = ['Accuracy','Precision_Micro', 'Recall_Micro', 'F1_Micro', 'Precision_Macro',
                        'Recall_Macro', 'F1_Macro']
    table2 = PrettyTable()
    table2.title = 'Overall'
    table2.field_names = table2_cols_name
    with open(os.path.join(save_path, 'overall_result.csv'), 'w+', encoding='utf-8') as f:
        data = ['{:.5f}'.format(str2float(cm.Overall_ACC)),
                '{:.5f}'.format(str2float(cm.PPV_Micro)),
                '{:.5f}'.format(str2float(cm.TPR_Micro)),
                '{:.5f}'.format(str2float(cm.F1_Micro)),
                '{:.5f}'.format(str2float(cm.PPV_Macro)),
                '{:.5f}'.format(str2float(cm.TPR_Macro)),
                '{:.5f}'.format(str2float(cm.F1_Macro)),
                ]

        table2.add_row(data)

        f.write(','.join(table2_cols_name) + '\n')
        f.write(','.join(data))
    print(table2)

    with open(os.path.join(save_path, 'result.txt'), 'w+', encoding='utf-8') as f:
        f.write(str(table1))
        f.write('\n')
        f.write(str(table2))


class Test_Metrice:
    def __init__(self, y_true, y_pred, class_num):
        self.y_true = y_true
        self.y_pred = y_pred
        self.class_num = class_num
        self.result = {i: {} for i in range(self.class_num)}
        self.metrice = ['PPV', 'TPR', 'AUC', 'AUPR', 'F1']
        self.metrice_name = ['Precision', 'Recall', 'AUC', 'F1', 'ACC']

    def __call__(self):
        cm = ConfusionMatrix(self.y_true, self.y_pred)
        for j in range(len(self.metrice)):
            for i in range(self.class_num):
                self.result[i][self.metrice_name[j]] = str2float(eval('cm.{}'.format(self.metrice[j]))[i])

        return self.result, cm

def plot_confusion_matrix(cm, classes, save_path, normalize=True, title='Confusion matrix', cmap=plt.cm.Blues, name='test'):
    plt.figure(figsize=(min(len(classes), 30), min(len(classes), 30)))
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    trained_classes = classes
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(name + title, fontsize=min(len(classes), 30))  # title font size
    tick_marks = np.arange(len(classes))
    plt.xticks(np.arange(len(trained_classes)), classes, rotation=90, fontsize=min(len(classes), 30)) # X tricks font size
    plt.yticks(tick_marks, classes, fontsize=min(len(classes), 30)) # Y tricks font size
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, np.round(cm[i, j], 2), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black", fontsize=min(len(classes), 30)) # confusion_matrix font size
    plt.ylabel('True label', fontsize=min(len(classes), 30)) # True label font size
    plt.xlabel('Predicted label', fontsize=min(len(classes), 30)) # Predicted label font size
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'confusion_matrix.png'), dpi=150)
    plt.show()

def save_confusion_matrix(cm, classes, save_path):
    str_arr = []
    for class_, cm_ in zip(classes, cm):
        str_arr.append('{},{}'.format(class_, ','.join(list(map(lambda x:'{:.4f}'.format(x), list(cm_))))))
    str_arr.append(' ,{}'.format(','.join(classes)))

    with open(os.path.join(save_path, 'confusion_matrix.csv'), 'w+') as f:
        f.write('\n'.join(str_arr))


def visual_predictions(y_true, y_pred, y_score, path, label, save_path):
    true_ids = (y_true == y_pred)
    with open(os.path.join(save_path, 'correct.csv'), 'w+') as f:
        f.write('path,true_label,pred_label,pred_score\n')
        f.write('\n'.join(['{},{},{},{:.4f}'.format(i, label[j], label[k], z) for i, j, k, z in
                           zip(path[true_ids], y_true[true_ids], y_pred[true_ids], y_score[true_ids])]))

    with open(os.path.join(save_path, 'incorrect.csv'), 'w+') as f:
        f.write('path,true_label,pred_label,pred_score\n')
        f.write('\n'.join(['{},{},{},{:.4f}'.format(i, label[j], label[k], z) for i, j, k, z in
                           zip(path[~true_ids], y_true[~true_ids], y_pred[~true_ids], y_score[~true_ids])]))


def visual_tsne(feature, y_true, path, labels, save_path):
    color_name_list = list(sorted(cnames.keys()))
    np.random.shuffle(color_name_list)
    tsne = TSNE(n_components=2)
    feature_tsne = tsne.fit_transform(feature)

    if len(labels) <= len(color_name_list):
        plt.figure(figsize=(8, 8))
        for idx, i in enumerate(labels):
            plt.scatter(feature_tsne[y_true == idx, 0], feature_tsne[y_true == idx, 1], label=i,
                        c=cnames[color_name_list[idx]])
        plt.legend(loc='best')
        plt.title('Tsne Visual')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'tsne.png'))

    with open(os.path.join(save_path, 'tsne.csv'), 'w+') as f:
        f.write('path,label,tsne_x,tsne_y\n')
        f.write('\n'.join(
            ['{},{},{:.0f},{:.0f}'.format(i, labels[j], k[0], k[1]) for i, j, k in zip(path, y_true, feature_tsne)]))

def plot_log(opt):
    logs = pd.read_csv(os.path.join(opt.save_path, 'train.log'))

    plt.figure(figsize=(10, 10))
    plt.subplot(2, 2, 1)
    plt.plot(logs['loss'], label='train')
    plt.plot(logs['test_loss'], label='val')
    try:
        plt.plot(logs['kd_loss'], label='kd')
    except:
        pass
    plt.legend()
    plt.title('loss')
    plt.xlabel('epoch')

    plt.subplot(2, 2, 2)
    plt.plot(logs['acc'], label='train')
    plt.plot(logs['test_acc'], label='val')
    plt.legend()
    plt.title('acc')
    plt.xlabel('epoch')

    plt.subplot(2, 2, 3)
    plt.plot(logs['mean_acc'], label='train')
    plt.plot(logs['test_mean_acc'], label='val')
    plt.legend()
    plt.title('mean_acc')
    plt.xlabel('epoch')

    plt.tight_layout()
    plt.savefig(r'{}/iterative_curve.png'.format(opt.save_path))

    plt.figure(figsize=(7, 5))
    plt.plot(logs['lr'])
    plt.title('learning rate')
    plt.xlabel('epoch')

    plt.tight_layout()
    plt.savefig(r'{}/learning_rate_curve.png'.format(opt.save_path))
