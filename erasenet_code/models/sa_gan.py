import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torch.autograd import Variable
from models.networks import get_pad, ConvWithActivation, DeConvWithActivation
from pdb import set_trace as stx


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


def img2photo(imgs):
    return ((imgs+1)*127.5).transpose(1,2).transpose(2,3).detach().cpu().numpy()

def visual(imgs):
    im = img2photo(imgs)
    Image.fromarray(im[0].astype(np.uint8)).show()

class NonLocalBlock(nn.Module):
    def __init__(self, channel):
        super(NonLocalBlock, self).__init__()
        self.inter_channel = channel // 2
        self.conv_phi = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1,padding=0, bias=False)
        self.conv_theta = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_g = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.conv_mask = nn.Conv2d(in_channels=self.inter_channel, out_channels=channel, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        # [N, C, H , W]
        b, c, h, w = x.size()
        # [N, C/2, H * W]
        x_phi = self.conv_phi(x).view(b, c, -1)
        # [N, H * W, C/2]
        x_theta = self.conv_theta(x).view(b, c, -1).permute(0, 2, 1).contiguous()
        x_g = self.conv_g(x).view(b, c, -1).permute(0, 2, 1).contiguous()
        # [N, H * W, H * W]
        mul_theta_phi = torch.matmul(x_theta, x_phi)
        mul_theta_phi = self.softmax(mul_theta_phi)
        # [N, H * W, C/2]
        mul_theta_phi_g = torch.matmul(mul_theta_phi, x_g)
        # [N, C/2, H, W]
        mul_theta_phi_g = mul_theta_phi_g.permute(0,2,1).contiguous().view(b,self.inter_channel, h, w)
        # [N, C, H , W]
        mask = self.conv_mask(mul_theta_phi_g)
        out = mask + x
        return out

class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, same_shape=True, **kwargs):
        super(Residual,self).__init__()
        self.same_shape = same_shape
        strides = 1 if same_shape else 2
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1,stride=strides)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        # self.conv2 = torch.nn.utils.spectral_norm(self.conv2)
        if not same_shape:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1,
            # self.conv3 = nn.Conv2D(channels, kernel_size=3, padding=1,
                                 stride=strides)
            # self.conv3 = torch.nn.utils.spectral_norm(self.conv3)
        self.batch_norm2d = nn.BatchNorm2d(out_channels)

    def forward(self,x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        if not self.same_shape:
            x = self.conv3(x)
        out = self.batch_norm2d(out + x)
        # out = out + x
        return F.relu(out)

class Encoder(nn.Module):
    def __init__(self, n_in_channel=3):
        super().__init__()
        self.conv1 = ConvWithActivation(3,32,kernel_size=4,stride=2,padding=1)
        self.conva = ConvWithActivation(32,32,kernel_size=3, stride=1, padding=1)
        self.convb = ConvWithActivation(32,64, kernel_size=4, stride=2, padding=1)
        self.res1 = Residual(64,64)
        self.res2 = Residual(64,64)
        self.res3 = Residual(64,128,same_shape=False)
        self.res4 = Residual(128,128)
        self.res5 = Residual(128,256,same_shape=False)
       # self.nn = ConvWithActivation(256, 512, 3, 1, dilation=2, padding=get_pad(64, 3, 1, 2))
        self.res6 = Residual(256,256)
        self.res7 = Residual(256,512,same_shape=False)
        self.res8 = Residual(512,512)
        self.conv2 = ConvWithActivation(512,512,kernel_size=1)

    def forward(self,x):
        x = self.conv1(x)   #3,256,256
        x = self.conva(x)   #32,128,128
        con_x1 = x          #32,128,128
        x = self.convb(x)   #64,64,64
        x = self.res1(x)    #64,64,64
        con_x2 = x          #64,64,64
        x = self.res2(x)    #64,64,64
        x = self.res3(x)    #128, 32, 32
        con_x3 = x          #128, 32, 32
        x = self.res4(x)    #128, 32, 32
        x = self.res5(x)    #256, 16, 16
        con_x4 = x          #256, 16, 16
        x = self.res6(x)    #256, 16, 16
        x = self.res7(x)     #512, 8, 8
        x = self.res8(x)     #512, 8, 8
        x = self.conv2(x)    #512, 8, 8

        return x,con_x1,con_x2,con_x3,con_x4

class Decoder(nn.Module):
    def __init__(self, n_in_channel=3):
        super().__init__()
        self.conv_o1 = nn.Conv2d(64,3,kernel_size=1)
        self.conv_o2 = nn.Conv2d(32,3,kernel_size=1)
        self.deconv1 = DeConvWithActivation(512,256,kernel_size=3,padding=1,stride=2)
        self.deconv2 = DeConvWithActivation(256*2,128,kernel_size=3,padding=1,stride=2)
        self.deconv3 = DeConvWithActivation(128*2,64,kernel_size=3,padding=1,stride=2)
        self.deconv4 = DeConvWithActivation(64*2,32,kernel_size=3,padding=1,stride=2)
        self.deconv5 = DeConvWithActivation(64,3,kernel_size=3,padding=1,stride=2)
        
        self.lateral_connection1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1, padding=0,stride=1),
            nn.Conv2d(256, 512, kernel_size=3, padding=1,stride=1),
            nn.Conv2d(512, 512, kernel_size=3, padding=1,stride=1),
            nn.Conv2d(512, 256, kernel_size=1, padding=0,stride=1),)
        self.lateral_connection2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1, padding=0,stride=1),
            nn.Conv2d(128, 256, kernel_size=3, padding=1,stride=1),
            nn.Conv2d(256, 256, kernel_size=3, padding=1,stride=1),
            nn.Conv2d(256, 128, kernel_size=1, padding=0,stride=1),)
        self.lateral_connection3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, padding=0,stride=1),
            nn.Conv2d(64, 128, kernel_size=3, padding=1,stride=1),
            nn.Conv2d(128, 128, kernel_size=3, padding=1,stride=1),
            nn.Conv2d(128, 64, kernel_size=1, padding=0,stride=1),)
        self.lateral_connection4 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=1, padding=0,stride=1),
            nn.Conv2d(32, 64, kernel_size=3, padding=1,stride=1),
            nn.Conv2d(64, 64, kernel_size=3, padding=1,stride=1),
            nn.Conv2d(64, 32, kernel_size=1, padding=0,stride=1),) 

    def forward(self,x,con_x1,con_x2,con_x3,con_x4):
        #downsample
        x = self.deconv1(x)   #256, 16, 16 
        x = torch.cat([self.lateral_connection1(con_x4), x], dim=1) #512, 16, 16
        x = self.deconv2(x)   #128, 32, 32
        x = torch.cat([self.lateral_connection2(con_x3), x], dim=1) #256, 16, 16
        x = self.deconv3(x)   #64, 64, 64
        xo1 = x               #64, 64, 64
        x = torch.cat([self.lateral_connection3(con_x2), x], dim=1) #128, 64, 64
        x = self.deconv4(x)   #32, 128, 128
        xo2 = x               #32, 128, 128
        x = torch.cat([self.lateral_connection4(con_x1), x], dim=1) #64, 128, 128
        #import pdb;pdb.set_trace()
        x = self.deconv5(x)   #3, 256, 256
        x_o1 = self.conv_o1(xo1) #64, 64, 64
        x_o2 = self.conv_o2(xo2) #32, 128, 12
        x_o_unet = x          ##3, 256, 256

        return x_o_unet,x_o2,x_o1

class UNet_n2n_un(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(UNet_n2n_un, self).__init__()

        self.en_block1 = nn.Sequential(
            nn.Conv2d(in_channels, 48, 3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Conv2d(48, 48, 3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.MaxPool2d(2))

        self.en_block2 = nn.Sequential(
            nn.Conv2d(48, 48, 3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.MaxPool2d(2))

        self.en_block3 = nn.Sequential(
            nn.Conv2d(48, 48, 3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.MaxPool2d(2))

        self.en_block4 = nn.Sequential(
            nn.Conv2d(48, 48, 3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.MaxPool2d(2))

        self.en_block5 = nn.Sequential(
            nn.Conv2d(48, 48, 3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            NonLocalBlock(48),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 48, 3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            NonLocalBlock(48),
            nn.Upsample(scale_factor=2, mode='nearest'))

        self.de_block1 = nn.Sequential(
            nn.Conv2d(96, 96, 3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            NonLocalBlock(96),
            nn.Conv2d(96, 96, 3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Upsample(scale_factor=2, mode='nearest'))

        self.de_block2 = nn.Sequential(
            nn.Conv2d(144, 96, 3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Conv2d(96, 96, 3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Upsample(scale_factor=2, mode='nearest'))

        self.de_block3 = nn.Sequential(
            nn.Conv2d(208, 96, 3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Conv2d(96, 96, 3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Upsample(scale_factor=2, mode='nearest'))

        self.de_block4 = nn.Sequential(
            nn.Conv2d(176, 96, 3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Conv2d(96, 96, 3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Upsample(scale_factor=2, mode='nearest'))

        self.de_block5 = nn.Sequential(
            nn.Conv2d(96 + in_channels, 64, 3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Conv2d(64, 32, 3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),

            nn.Conv2d(32, out_channels, 3, padding=1, bias=True))
        

        # Initialize weights
        # self._init_weights()

    # def _init_weights(self):
    #     """Initializes weights using He et al. (2015)."""
    #     for m in self.modules():
    #         if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
    #             nn.init.kaiming_normal_(m.weight.data)
    #             m.bias.data.zero_()

    def forward(self, x,con_x2,con_x1):
        pool1 = self.en_block1(x)
        pool2 = self.en_block2(pool1)
        pool3 = self.en_block3(pool2)
        pool4 = self.en_block4(pool3)
        upsample5 = self.en_block5(pool4)
        concat5 = torch.cat((upsample5, pool4), dim=1)
        upsample4 = self.de_block1(concat5)
        concat4 = torch.cat((upsample4, pool3), dim=1)
        upsample3 = self.de_block2(concat4)
        concat3 = torch.cat((upsample3, pool2,con_x2), dim=1)
        upsample2 = self.de_block3(concat3)
        concat2 = torch.cat((upsample2, pool1,con_x1), dim=1)
        upsample1 = self.de_block4(concat2)
        concat1 = torch.cat((upsample1, x), dim=1)
        out = self.de_block5(concat1)

        return out



class STRnet2(nn.Module):
    def __init__(self, n_in_channel=3):
        super(STRnet2, self).__init__()
        self.UnetEncoder=Encoder(n_in_channel) 
#         for parm in self.UnetEncoder.parameters():
#             parm.requires_grad=False
        self.UnetDecoder=Decoder(n_in_channel)
#         for parm in self.UnetDecoder.parameters():
#             parm.requires_grad=False
        self.NafNet=UNet_n2n_un(n_in_channel)
    
    def forward(self, x):
        
        x,con_x1,con_x2,con_x3,con_x4=self.UnetEncoder(x)
        
        x_o_unet,x_o2,x_o1=self.UnetDecoder(x,con_x1,con_x2,con_x3,con_x4)
        
        output=self.NafNet(x_o_unet,con_x2,con_x1)

        return output,x_o_unet,x_o2,x_o1




if __name__ == "__main__":
    img_channel = 3
 
    # using('start . ')
    model =STRnet2(img_channel)
    model.eval()
    # print(model)
    input = torch.randn(4, 3, 256, 256)
    # input = torch.randn(1, 3, 32, 32)
    y = model(input)
    print(y.size())
   
