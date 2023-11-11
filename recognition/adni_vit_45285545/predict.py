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
    '''
    Runs the given model on the given image and returns a predicted label.
    Applies the same pre-processing as validation and testing procedures.
    '''
    # Pre-process image by converting to float tensor and centre-cropping to 224
    image = transforms.ToTensor()(image).unsqueeze(0).to(device)
    image = transforms.CenterCrop(224)(image)
    image = transforms.ConvertImageDtype(torch.float)(image)

    # Use the model to run inference on the image and get the model prediction
    output = mdl(image)
    _, predicted = torch.max(output.data, 1)

    # Return a string label corresponding to the model prediction
    label = {0: 'Cognitive normal', 1: 'Alzheimer\'s disease'}[int(predicted[0])]
    return 'Predicted: ' + label


def main(args):
    # Load the model stored at the given filepath; onto CUDA by default
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mdl = torch.load(args.mdlfile).to(device)

    # If "--test" flag specified, run model on ADNI test split and output results
    if args.test:
        test_model(mdl, device=device, agg=args.agg, pg=True)

    # If "--gui" flag specified, launch gradio GUI for interactive model use
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
