import warnings
warnings.filterwarnings("ignore")
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os, argparse, shutil, random, imp
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from dataset import *
from modules import *
import torch, torchvision, time, datetime, copy
from utils.utilis_ import update_opt,show_config,dict_to_PrettyTable,select_device,\
    setting_optimizer,load_weights,WarmUpLR, save_model,plot_log
from utils.utils_loss import *
from copy import deepcopy
from utils.utils_fit import fitting



torch.backends.cudnn.deterministic = True
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='vit_base_patch16_224', help='model name')
    parser.add_argument('--pretrained', action="store_true", help='using pretrain weight')
    parser.add_argument('--weight', type=str, default='', help='loading weight path')
    parser.add_argument('--config', type=str, default='config/config.py', help='config path')
    parser.add_argument('--device', type=str, default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')

    parser.add_argument('--train_path', type=str, default=r'dataset/train', help='train data path')
    parser.add_argument('--val_path', type=str, default=r'dataset/val', help='val data path')
    parser.add_argument('--test_path', type=str, default=r'dataset/test', help='test data path')
    parser.add_argument('--label_path', type=str, default=r'dataset/label.txt', help='label path')
    parser.add_argument('--image_size', type=int, default=224, help='image size')
    parser.add_argument('--image_channel', type=int, default=1, help='image channel')
    parser.add_argument('--workers', type=int, default=4, help='dataloader workers')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size (-1 for autobatch)')
    parser.add_argument('--epoch', type=int, default=2, help='epoch')
    parser.add_argument('--save_path', type=str, default=r'runs/exp', help='save path for model and log')
    parser.add_argument('--resume', action="store_true", help='resume from save_path traning')

    # optimizer parameters
    parser.add_argument('--loss', type=str, choices=['PolyLoss', 'CrossEntropyLoss', 'FocalLoss'],
                        default='CrossEntropyLoss', help='loss function')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'AdamW', 'RMSProp'], default='AdamW', help='optimizer')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='label smoothing')
    parser.add_argument('--class_balance', action="store_true", help='using class balance in loss')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight_decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum in optimizer')
    parser.add_argument('--amp', action="store_true", help='using AMP(Automatic Mixed Precision)')
    parser.add_argument('--warmup', action="store_true", help='using WarmUp LR')
    parser.add_argument('--warmup_ratios', type=float, default=0.05,
                        help='warmup_epochs = int(warmup_ratios * epoch) if warmup=True')
    parser.add_argument('--warmup_minlr', type=float, default=1e-6,
                        help='minimum lr in warmup(also as minimum lr in training)')
    parser.add_argument('--metrice', type=str, choices=['loss', 'acc', 'mean_acc'], default='acc', help='best.pt save relu')
    parser.add_argument('--patience', type=int, default=30, help='EarlyStopping patience (--metrice without improvement)')

    # Data Processing parameters
    parser.add_argument('--imagenet_meanstd', action="store_true", help='using ImageNet Mean and Std')
    parser.add_argument('--Augment', type=str,
                        choices=['RandAugment', 'AutoAugment', 'TrivialAugmentWide', 'AugMix', 'none'], default='none',
                        help='Data Augment')

    opt = parser.parse_known_args()[0]

    # load the parameter for resuming the training phase
    if opt.resume:
        opt.resume = True
        if not os.path.exists(os.path.join(opt.save_path, 'last.pt')):
            raise Exception('last.pt not found. please check your --save_path folder and --resume parameters')
        ckpt = torch.load(os.path.join(opt.save_path, 'last.pt'))
        opt = ckpt['opt']
        opt.resume = True
        print('found checkpoint from {}, model type:{}\n{}'.format(opt.save_path, ckpt['model'].name, dict_to_PrettyTable(ckpt['best_metrice'], 'Best Metrice')))
    else:
        if os.path.exists(opt.save_path):
            shutil.rmtree(opt.save_path)
        os.makedirs(opt.save_path)
        config = imp.load_source('config', opt.config).Config()
        shutil.copy(__file__, os.path.join(opt.save_path, 'main.py'))
        shutil.copy(opt.config, os.path.join(opt.save_path, 'config.py'))
        opt = update_opt(opt, config._get_opt())

    set_seed(opt.random_seed)
    show_config(deepcopy(opt))

    CLASS_NUM = len(os.listdir(opt.train_path))
    DEVICE = select_device(opt.device, opt.batch_size)

    # load_data, transformation and aug
    train_transform, val_transform = get_transform(opt)
    train_data ,class_weight= preprocessing_train_loader(opt, train_transform)
    val_data = preprocessing_val_loader(opt, val_transform)



    if opt.resume:
        model = ckpt['model'].to(DEVICE).float()
    else:
        model = load_model(opt,CLASS_NUM)
        model = load_weights(model, opt).to(DEVICE)


    scaler = torch.cuda.amp.GradScaler(enabled=(opt.amp if torch.cuda.is_available() else False))
    optimizer = setting_optimizer(opt, model)
    lr_scheduler = WarmUpLR(optimizer, opt)

    if opt.resume:
        optimizer.load_state_dict(ckpt['optimizer'])
        lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
        loss = ckpt['loss'].to(DEVICE)
        scaler.load_state_dict(ckpt['scaler'])

    else:
        loss = eval(opt.loss)(label_smoothing=opt.label_smoothing,
                              weight=torch.from_numpy(class_weight).to(DEVICE).float())

    return opt, model, train_data, val_data, optimizer, scaler, lr_scheduler, loss, DEVICE, CLASS_NUM, (
        ckpt['epoch'] if opt.resume else 0), (ckpt['best_metrice'] if opt.resume else None)





if __name__ == '__main__':
    opt, model, train_dataset, val_dataset, optimizer, scaler, \
        lr_scheduler, loss, DEVICE, CLASS_NUM, begin_epoch, best_metrice = parse_opt()

    if not opt.resume:
        save_epoch = 0
        with open(os.path.join(opt.save_path, 'train.log'), 'w+') as f:
            f.write('epoch,lr,loss,acc,mean_acc,test_loss,test_acc,test_mean_acc')
    else:
        save_epoch = torch.load(os.path.join(opt.save_path, 'last.pt'))['best_epoch']

    print('{} begin train!'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

    for epoch in range(begin_epoch, opt.epoch):
        if epoch > (save_epoch + opt.patience) and opt.patience != 0:
            print('No Improve from {} to {}, EarlyStopping.'.format(save_epoch + 1, epoch))
            break

        begin = time.time()
        # training and evaluate
        metrice = fitting(model, loss, optimizer, train_dataset, val_dataset, CLASS_NUM,
                          DEVICE, scaler,'{}/{}'.format(epoch + 1, opt.epoch), opt)

        # save parameters
        with open(os.path.join(opt.save_path, 'train.log'), 'a+') as f:
            f.write(
                '\n{},{:.10f},{}'.format(epoch + 1, optimizer.param_groups[2]['lr'], metrice[1]))

        n_lr = optimizer.param_groups[2]['lr']
        lr_scheduler.step()

        # store the parameter of best epoch
        if best_metrice is None:
            best_metrice = metrice[0]
            save_model(
                os.path.join(opt.save_path, 'best.pt'),
                **{
                    'model': deepcopy(model).to('cpu').half(),
                    'opt': opt,
                    'best_metrice': best_metrice,
                }
            )
            save_epoch = epoch
        else:
            if eval('{} {} {}'.format(metrice[0]['test_{}'.format(opt.metrice)], '<' if opt.metrice == 'loss' else '>', best_metrice['test_{}'.format(opt.metrice)])):
                best_metrice = metrice[0]
                save_model(
                    os.path.join(opt.save_path, 'best.pt'),
                    **{
                    'model': deepcopy(model).to('cpu').half(),
                    'opt': opt,
                    'best_metrice': best_metrice,
                    }
                )
                save_epoch = epoch
        # save the parameter for last epoch
        save_model(
            os.path.join(opt.save_path, 'last.pt'),
            **{
               'model': deepcopy(model).to('cpu').half(),
               'opt': opt,
               'epoch': epoch + 1,
               'optimizer' : optimizer.state_dict(),
               'lr_scheduler': lr_scheduler.state_dict(),
               'best_metrice': best_metrice,
               'loss': deepcopy(loss).to('cpu'),
               'scaler': scaler.state_dict(),
               'best_epoch': save_epoch,
            }
        )
    #
        print(dict_to_PrettyTable(metrice[0], '{} epoch:{}/{}, best_epoch:{}, time:{:.2f}s, lr:{:.8f}'.format(
                    datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    epoch + 1, opt.epoch, save_epoch + 1, time.time() - begin, n_lr,
                )))
    #
    plot_log(opt)

