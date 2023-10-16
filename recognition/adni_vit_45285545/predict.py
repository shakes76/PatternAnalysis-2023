'''
Example usage of ViT model trained on ADNI dataset.
'''
import argparse

from typing import Any

import gradio as gr
import torch

from PIL import Image
from torchvision import transforms

from train import test_model


def predict(image: Image, mdl: Any, device: torch.device) -> int:
    '''Runs the given model on the given image and returns a predicted label.'''
    image = transforms.ToTensor()(image).unsqueeze(0).to(device)
    image = transforms.ConvertImageDtype(torch.float)(image)
    output = mdl(image)
    _, predicted = torch.max(output.data, 1)
    label = {0: 'Cognitive normal', 1: 'Alzheimer\'s disease'}[int(predicted[0])]
    return 'Predicted: ' + label


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mdl = torch.load(args.mdlfile).to(device)

    if args.test:
        test_model(mdl, device=device, agg=args.agg, pg=True)

    if args.gui:
        mdlpredict = lambda image: predict(image, mdl, device)
        app = gr.Interface(fn=mdlpredict, inputs='image', outputs='text')
        app.launch()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mdlfile', help='trained model file (.pt)')
    parser.add_argument('--agg', action='store_true', help='aggregate test results by patient')
    parser.add_argument('--test', action='store_true', help='run inference ADNI test split')
    parser.add_argument('--gui', action='store_true', help='start interactive web GUI')
    args = parser.parse_args()
    main(args)
