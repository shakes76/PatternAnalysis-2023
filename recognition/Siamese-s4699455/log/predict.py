import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image

from nets.modules import Siamese as siamese
from utils.utils import letterbox_image, preprocess_input, cvtColor, show_config


class Siamese(object):
    _defaults = {
        "model_path"        : 'logs/best_epoch_weights.pth',
        "input_shape"       : [224, 208],
        "letterbox_image"   : False,
        "cuda"              : True
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
        self.generate()
        
        show_config(**self._defaults)
        
    def generate(self):
        print('Loading weights into state dict...')
        device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model   = siamese(self.input_shape)
        print(model)
        model.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net = model.eval()
        print('{} model loaded.'.format(self.model_path))

        if self.cuda:
            self.net = torch.nn.DataParallel(self.net)
            cudnn.benchmark = True
            self.net = self.net.cuda()
    
    def letterbox_image(self, image, size):
        image   = image.convert("RGB")
        iw, ih  = image.size
        w, h    = size
        scale   = min(w/iw, h/ih)
        nw      = int(iw*scale)
        nh      = int(ih*scale)

        image       = image.resize((nw,nh), Image.BICUBIC)
        new_image   = Image.new('RGB', size, (128,128,128))
        new_image.paste(image, ((w-nw)//2, (h-nh)//2))
        if self.input_shape[-1]==1:
            new_image = new_image.convert("L")
        return new_image
        
    def detect_image(self, image_1, image_2):
        photo_1  = preprocess_input(np.array(image_1, np.float32))
        photo_1 = photo_1[16:-16,16:-16]
        photo_2  = preprocess_input(np.array(image_2, np.float32))
        photo_2 = photo_2[16:-16,16:-16]

        with torch.no_grad():
            photo_1 = np.expand_dims(photo_1,axis=2)
            # print(image.shape)
            photo_1 = np.transpose(photo_1, [2, 1, 0]) #[2, 0, 1]
            photo_1 = np.expand_dims(photo_1,axis=0) 
            photo_1 = torch.from_numpy(photo_1)

            photo_2 = np.expand_dims(photo_2,axis=2) 
            # print(image.shape)
            photo_2 = np.transpose(photo_2, [2, 1, 0]) #[2, 0, 1]
            photo_2 = np.expand_dims(photo_2,axis=0) 
            photo_2 = torch.from_numpy(photo_2)


            # photo_1 = torch.from_numpy(np.expand_dims(np.transpose(photo_1, (2, 1, 0)), 0)).type(torch.FloatTensor)
            # photo_2 = torch.from_numpy(np.expand_dims(np.transpose(photo_2, (2, 1, 0)), 0)).type(torch.FloatTensor)
            
            if self.cuda:
                photo_1 = photo_1.cuda()
                photo_2 = photo_2.cuda()
                
            # output = self.net([photo_1, photo_2])[0]
            output = self.net(photo_1, photo_2)[0]
            output = torch.nn.Sigmoid()(output)

        plt.subplot(1, 2, 1)
        plt.imshow(np.array(image_1))

        plt.subplot(1, 2, 2)
        plt.imshow(np.array(image_2))
        plt.text(-12, -12, 'Similarity:%.3f' % output, ha='center', va= 'bottom',fontsize=11)
        
        plt.savefig('./out__'+str(output.tolist())+'.png')
        plt.show()
        return output
