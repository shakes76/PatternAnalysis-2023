import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)
    
class ContextLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ContextLayer, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Dropout3d(p=0.3),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)
    
class LocalizationLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LocalizationLayer, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1,  stride=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)
    
class UNET(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=1, features=[64, 128, 256, 512],
    ):
        super(UNET, self).__init__()
        self.feature_num = 64
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        #DoubleConv(self.feature_num*8, self.feature_num*16)
        #LAYER 1
        self.first_conv = nn.Conv2d(in_channels, self.feature_num, kernel_size=3, stride=1, padding=1, bias=False)
        self.first_down = ContextLayer(self.feature_num, self.feature_num)

        #LAYER 2
        self.second_conv = nn.Conv2d(self.feature_num, self.feature_num*2, kernel_size=3, stride=2, padding=1, bias=False)
        self.second_down = ContextLayer(self.feature_num*2, self.feature_num*2)

        #LAYER 3
        self.third_conv = nn.Conv2d(self.feature_num*2, self.feature_num*4, kernel_size=3, stride=2, padding=1, bias=False)
        self.third_down = ContextLayer(self.feature_num*4, self.feature_num*4)

        #LAYER 4
        self.fourth_conv = nn.Conv2d(self.feature_num*4, self.feature_num*8, kernel_size=3, stride=2, padding=1, bias=False)
        self.fourth_down = ContextLayer(self.feature_num*8, self.feature_num*8)

        #LAYER 5
        self.fifth_conv = nn.Conv2d(self.feature_num*8, self.feature_num*16, kernel_size=3, stride=2, padding=1, bias=False)
        self.fifth_down = ContextLayer(self.feature_num*16, self.feature_num*16)

        #LAYER -5
        self.first_upsam = nn.ConvTranspose2d(self.feature_num*16, self.feature_num*8, kernel_size=2, stride=2)

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.first_upsample_d1 = nn.Conv2d(self.feature_num*16,self.feature_num*8,kernel_size=1, stride = 1)

        #LAYER -4
        self.first_local = LocalizationLayer(self.feature_num*16, self.feature_num*8)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.second_upsample_d1 = nn.Conv2d(self.feature_num*8,self.feature_num*4,kernel_size=1, stride = 1)        

        #LAYER -3
        self.second_local = LocalizationLayer(self.feature_num*8, self.feature_num*4)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.third_upsample_d1 = nn.Conv2d(self.feature_num*4,self.feature_num*2,kernel_size=1, stride = 1)        

        #LAYER -2
        self.third_local = LocalizationLayer(self.feature_num*4, self.feature_num*2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.fourth_upsample_d1 = nn.Conv2d(self.feature_num*2,self.feature_num*1,kernel_size=1, stride = 1)        

        #LAYER -1
        self.final_conv_layer = DoubleConv(self.feature_num*2, self.feature_num*2)
        self.final_activation = nn.Sigmoid()
        
        self.segmentation_1 = nn.Conv2d(self.feature_num*4, out_channels, kernel_size=1)
        self.segmentation_2 = nn.Conv2d(self.feature_num*2, out_channels, kernel_size=1)
        self.segmentation_3 = nn.Conv2d(self.feature_num*2, out_channels, kernel_size=1)

    def forward(self, x):
        #LAYER 1
        l1_x1 = self.first_conv(x)
        l1_x2 = self.first_down(l1_x1)
        x = torch.add(l1_x2, l1_x1) 
        skip_connections1 = x     
           

        #LAYER 2
        l2_x1 = self.second_conv(x)
        l2_x2 = self.second_down(l2_x1)
        x = torch.add(l2_x2, l2_x1) 
        skip_connections2 = x    

        #LAYER 3
        l3_x1 = self.third_conv(x)
        l3_x2 = self.third_down(l3_x1)
        x = torch.add(l3_x2, l3_x1) 
        skip_connections3 = x  

        #LAYER 4
        l4_x1 = self.fourth_conv(x)
        l4_x2 = self.fourth_down(l4_x1)
        x = torch.add(l4_x2, l4_x1) 
        skip_connections4 = x  

        #LAYER 5
        l5_x1 = self.fifth_conv(x)
        l5_x2 = self.fifth_down(l5_x1)
        x = torch.add(l5_x2, l5_x1) 
        x = self.upsample(x)
        x = self.first_upsample_d1(x)

        #LAYER -4
        concat_skip = torch.cat((skip_connections4, x), dim=1)
        x = self.first_local(concat_skip)
        x = self.upsample(x)
        x = self.second_upsample_d1(x)   

        #LAYER -3
        concat_skip = torch.cat((skip_connections3, x), dim=1)
        x = self.second_local(concat_skip)
        self.segment_1 = self.segmentation_1(x)
        self.segment_1_upscaled = self.upsample(self.segment_1)
        x = self.upsample(x)
        x = self.third_upsample_d1(x) 

        #LAYER -2
        concat_skip = torch.cat((skip_connections2, x), dim=1)
        x = self.third_local(concat_skip)
        self.segment_2 = self.segmentation_2(x)
        self.segment_1_2 = torch.cat((self.segment_1_upscaled, self.segment_2), dim=1)
        self.segment_1_2_upscaled = self.upsample(self.segment_1_2)
        x = self.upsample(x)
        x = self.fourth_upsample_d1(x) 

        #LAYER -1
        concat_skip = torch.cat((skip_connections1, x), dim=1)
        x = self.final_conv_layer(concat_skip)
        self.segment_3 = self.segmentation_3(x)
        self.segment_1_2_3 = torch.cat((self.segment_1_2_upscaled, self.segment_3), dim=1)
        x = self.final_activation(self.segment_1_2_3)
        return x

def test():
    x = torch.randn((3, 1, 161, 161))
    model = UNET(in_channels=1, out_channels=1)
    preds = model(x)
    assert preds.shape == x.shape

if __name__ == "__main__":
    test()