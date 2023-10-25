import torch
import torch.nn as nn

def image_patcher(image,size_patch, patch_depth):
    Batch_Size, Depth, C, Height, Width = image.shape
    # change the shape of the tensor
    Height_final = Height // size_patch
    Width_final = Width // size_patch
    Depth_final = Depth // patch_depth
    image = image.reshape(Batch_Size, Depth_final, patch_depth, C,Height_final,size_patch,Width_final,size_patch)
    #permute the dimensions of the tensor
    image = image.permute(0, 1, 4, 6, 3, 2, 5, 7)  
    #flatten specific dimensions of the tensor
    image = image.flatten(1, 3).flatten(2, 5)
    return image

class AttentionBlock(nn.Module):
    def __init__(self,input_dimen,hiddenlayer_dimen,number_heads):
        super().__init__()
        #layer normalization is applied to the input data 
        self.input_layer_norm = nn.LayerNorm(input_dimen)
        #normalizes the output of the attention mechanism.
        self.output_layer_norm = nn.LayerNorm(input_dimen)
        # block with multiple attention heads.
        self.multihead_attention = nn.MultiheadAttention(input_dimen,number_heads)
        self.linear = nn.Sequential(nn.Linear(input_dimen,hiddenlayer_dimen),nn.GELU(),
            nn.Linear(hiddenlayer_dimen,input_dimen),
        )

    def forward(self,image):
        inp_x = self.input_layer_norm(image)
        add = self.multihead_attention(inp_x, inp_x, inp_x)[0]
        image = image + add
        image = image + self.linear(self.output_layer_norm(image))
        return image
    
class VisionTransformer(nn.Module):
    def __init__(
        self,input_dimen,hiddenlayer_dimen,number_heads,transform_layers,predict_num,size_patch
    ):
        super().__init__()
        (size_patch_x, size_patch_y) = size_patch

        self.size_patch = size_patch_x * size_patch_y
        #creates an instance of the nn.linear
        self.input_layer = nn.Linear(5*self.size_patch, input_dimen)
        #creates an instance of nn.sequential
        self.final_transform = nn.Sequential(*(AttentionBlock(input_dimen, hiddenlayer_dimen, number_heads) for _ in range(transform_layers)))

        self.dense_head = nn.Sequential(nn.LayerNorm(input_dimen), nn.Linear(input_dimen, predict_num))
        final_num_patch = 1 + (240 // size_patch_x)*(256 // size_patch_y)
        self.positional_emb = nn.Parameter(torch.randn(1,4*final_num_patch,input_dimen))
        self.classification_tkn = nn.Parameter(torch.randn(1,1,input_dimen))


    def forward(self, image):
        # input being preprocessed
        image = image_patcher(image, 16, 5)
        Batch_Size, x, _ = image.shape

        image = self.input_layer(image)

        # Add a positional encoding and a CLS token 
        classification_tkn = self.classification_tkn.repeat(Batch_Size, 1, 1)
        image = torch.cat([classification_tkn, image], dim=1)
        image = image + self.positional_emb[:, : x + 1]

        #this adds a final_transform
        image = image.transpose(0, 1)
        image = self.final_transform(image)
        class_ = image[0]
        out = self.dense_head(class_)
        return out