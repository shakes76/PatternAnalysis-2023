import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data


"""Autoregressive PixelCNN model"""
class PixelCNN(nn.Module):
    
    def __init__(self, in_channels, hidden_channels, num_embeddings):
        super(PixelCNN, self).__init__()
        # Equal to the number of embeddings in the VQVAE
        self.num_embeddings = num_embeddings
        # Initial convolutions skipping the center pixel
        self.conv_vstack = VerticalConv(in_channels, hidden_channels, mask_center=True)
        self.conv_hstack = HorizontalConv(in_channels, hidden_channels, mask_center=True)
        # Convolution block of PixelCNN. Uses dilation instead of downscaling
        self.conv_layers = nn.ModuleList([
            GatedMaskedConv(hidden_channels),
            GatedMaskedConv(hidden_channels, dilation=2),
            GatedMaskedConv(hidden_channels),
            GatedMaskedConv(hidden_channels, dilation=4),
            GatedMaskedConv(hidden_channels),
            GatedMaskedConv(hidden_channels, dilation=2),
            GatedMaskedConv(hidden_channels)
        ])
        # Output classification convolution (1x1)
        # The output channels should be in_channels*number of embeddings to learn continuous space and calc. CrossEntropyLoss
        self.conv_out = nn.Conv2d(hidden_channels, in_channels*self.num_embeddings, kernel_size=1, padding=0)
        
        
    def forward(self, x):
        # Scale input from 0 to 255 to -1 to 1
        x = (x.float() / 255.0) * 2 - 1

        # Initial convolutions
        v_stack = self.conv_vstack(x)
        h_stack = self.conv_hstack(x)
        # Gated Convolutions
        for layer in self.conv_layers:
            v_stack, h_stack = layer(v_stack, h_stack)
        # 1x1 classification convolution
        # Apply ELU (exponential activation function) before 1x1 convolution for non-linearity on residual connection
        out = self.conv_out(F.elu(h_stack))

        # Output dimensions: [Batch, Classes, Channels, Height, Width] (classes = num_embeddings)
        out = out.reshape(out.shape[0], self.num_embeddings, out.shape[1]//256, out.shape[2], out.shape[3])
        return out

    """Indices shape should be in form B C H W
    Pixels to fill should be marked with -1"""
    @torch.no_grad()
    def sample(self, ind_shape, ind):
        # Generation loop (iterating through pixels across channels)
        for h in range(ind_shape[2]):                   # Heights
            for w in range(ind_shape[3]):               # Widths
                for c in range(ind_shape[1]):           # Channels
                    # Skip if not to be filled (-1)
                    if (ind[:,c,h,w] != -1).all().item():
                        continue
                    # Only have to input upper half of ind (rest are masked anyway)
                    pred = self.forward(ind[:,:,:h+1,:])
                    probs = F.softmax(pred[:,:,c,h,w], dim=-1)
                    ind[:,c,h,w] = torch.multinomial(probs, num_samples=1).squeeze(dim=-1)
        return ind


"""A general Masked convolution, with a the mask as a parameter."""
class MaskedConvolution(nn.Module):

    def __init__(self, in_channels, out_channels, mask, dilation=1):
        
        super(MaskedConvolution, self).__init__()
        kernel_size = (mask.shape[0], mask.shape[1])
        padding = ([dilation*(kernel_size[i] - 1) // 2 for i in range(2)])
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)

        # Mask as buffer (must be moved with devices)
        self.register_buffer('mask', mask[None,None])

    def forward(self, x):
        self.conv.weight.data *= self.mask               # Set all following weights to 0 (make sure it is in GPU)
        return self.conv(x)


class VerticalConv(MaskedConvolution):
    # Masks all pixels below
    def __init__(self, in_channels, out_channels, kernel_size=3, mask_center=False, dilation=1):
        mask = torch.ones(kernel_size, kernel_size)
        mask[kernel_size//2+1:,:] = 0
        # For the first convolution, mask center row
        if mask_center:
            mask[kernel_size//2,:] = 0

        super().__init__(in_channels, out_channels, mask, dilation=dilation)

class HorizontalConv(MaskedConvolution):

    def __init__(self, in_channels, out_channels, kernel_size=3, mask_center=False, dilation=1):
        # Mask out all pixels on the left. (Note that kernel has a size of 1
        # in height because we only look at the pixel in the same row)
        mask = torch.ones(1,kernel_size)
        mask[0,kernel_size//2+1:] = 0

        # For first convolution, mask center pixel
        if mask_center:
            mask[0,kernel_size//2] = 0

        super().__init__(in_channels, out_channels, mask, dilation=dilation)
        
"""Gated Convolutions Model"""
class GatedMaskedConv(nn.Module):

    def __init__(self, in_channels, dilation=1):

        super(GatedMaskedConv, self).__init__()
        self.conv_vert = VerticalConv(in_channels, out_channels=2*in_channels, dilation=dilation)
        self.conv_horiz = HorizontalConv(in_channels, out_channels=2*in_channels, dilation=dilation)
        self.conv_vert_to_horiz = nn.Conv2d(2*in_channels, 2*in_channels, kernel_size=1, padding=0)
        self.conv_horiz_1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0)

    def forward(self, v_stack, h_stack):
        # Vertical stack (left)
        v_stack_feat = self.conv_vert(v_stack)
        v_val, v_gate = v_stack_feat.chunk(2, dim=1)
        v_stack_out = torch.tanh(v_val) * torch.sigmoid(v_gate)

        # Horizontal stack (right)
        h_stack_feat = self.conv_horiz(h_stack)
        h_stack_feat = h_stack_feat + self.conv_vert_to_horiz(v_stack_feat)
        h_val, h_gate = h_stack_feat.chunk(2, dim=1)
        h_stack_feat = torch.tanh(h_val) * torch.sigmoid(h_gate)
        h_stack_out = self.conv_horiz_1x1(h_stack_feat)
        h_stack_out = h_stack_out + h_stack

        return v_stack_out, h_stack_out