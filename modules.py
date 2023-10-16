import torch
import torch.nn as nn


class ADaIN(nn.Module):
    def __init__(self, epsilon=1e-5):
        super(ADaIN, self).__init__()
        self.epsilon = epsilon

    def forward(self, content, style):
        # Calculating mean and std from content and style tensors
        # Dim denoted as 2nd and 3rd dimension for height and width respectively
        content_mean = content.mean(dim=[2, 3], keepdim=True)
        content_std = content.std(dim=[2, 3], keepdim=True)

        style_mean = style.mean(dim=[2, 3], keepdim=True)
        style_std = style.std(dim=[2, 3], keepdim=True)

        # Normalizing the content tensor using the statistics from content tensor
        normalized_content = (content - content_mean) / (content_std + self.epsilon)

        # Scale and shift normalized content
        stylized_content = style_std * normalized_content + style_mean

        return stylized_content
