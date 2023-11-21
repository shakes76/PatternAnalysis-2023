import torch
import torch.nn as nn
import model as models
from thop import clever_format, profile

def select_model(name, num_classes, input_shape=None, channels=None, pretrained=False):
    if  name.startswith('vit_'):
        model = eval('models.{}(pretrained={})'.format(name, pretrained))
        model.head = nn.Linear(model.head.in_features, num_classes)
    elif name.startswith('resnet'):
        model = eval('models.{}(pretrained={})'.format(name, pretrained))
        model.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(model.fc.in_features, num_classes)
        )

    else:
        raise 'Unsupported Model Name.'

    if input_shape and channels:
    # calculate parameter values and flops
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dummy_input = torch.randn(1, channels, input_shape[0], input_shape[1]).to(device)
        flops, params = profile(model.to(device), (dummy_input,), verbose=False)

        # flops = flops * 2
        flops, params = clever_format([flops, params], "%.3f")
        print('Select Model: {}'.format(name))
        print('Total FLOPS: %s' % (flops))
        print('Total params: %s' % (params))
    model.name = name
    return model

if __name__ == '__main__':
    pass