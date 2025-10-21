import torch
import torch.nn as nn
import torch.nn.functional as F
from .Transformer import Transformer

class REBNCONV(nn.Module):
    def __init__(self,in_ch=3,out_ch=3,dirate=1):
        super(REBNCONV,self).__init__()

        self.conv_s1 = nn.Conv2d(in_ch,out_ch,3,padding=1*dirate,dilation=1*dirate)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self,x):

        hx = x
        xout = self.relu_s1(self.bn_s1(self.conv_s1(hx)))

        return xout

## upsample tensor 'src' to have the same spatial size with tensor 'tar'
def _upsample_like(src,tar):

    src = F.upsample(src,size=tar.shape[2:],mode='bilinear')

    return src

class Nested_Trans_U(nn.Module):

    def __init__(self, in_ch=128, mid_ch=64,num_patches=144):
        super(Nested_Trans_U,self).__init__()
        self.mid_ch = mid_ch
        self.rebnconvin  = nn.Sequential(nn.Conv2d(in_ch, 64, kernel_size=3, padding=1, bias=True),
                      nn.BatchNorm2d(64),
                      nn.LeakyReLU(inplace=True),
                                         )
        self.rebnconv1 = nn.Sequential(nn.Conv2d(256, 64, kernel_size=3, padding=1, bias=True),
                      nn.BatchNorm2d(64),
                      nn.LeakyReLU(inplace=True),
                                       )
        self.rebnconv2 = nn.Sequential(nn.Conv2d(256, 64, kernel_size=3, padding=1, bias=True),
                      nn.BatchNorm2d(64),
                      nn.LeakyReLU(inplace=True),
                                       )
        self.rebnconv3_1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True),
                      nn.BatchNorm2d(64),
                      nn.LeakyReLU(inplace=True),
                                         )
        self.global_attn = Transformer(depth=2, num_heads=1, embed_dim=self.mid_ch, mlp_ratio=3, num_patches=num_patches)

        self.rebnconv3_2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True),
                      nn.BatchNorm2d(64),
                      nn.LeakyReLU(inplace=True),
                                         )

        self.rebnconv2d = nn.Sequential(nn.Conv2d(mid_ch*2,mid_ch*4, kernel_size=3, padding=1, bias=True),
                      nn.BatchNorm2d(mid_ch*4),
                      nn.LeakyReLU(inplace=True),
                                        )

        self.rebnconv1d = nn.Sequential(nn.Conv2d(mid_ch*2,mid_ch*4, kernel_size=3, padding=1, bias=True),
                      nn.BatchNorm2d(mid_ch*4),
                      nn.LeakyReLU(inplace=True),
                                        )

        self.rebnconvout  = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True),
                      nn.BatchNorm2d(64),
                      nn.LeakyReLU(inplace=True))


    def forward(self,x):
        hxin = self.rebnconvin(x) # [B,64,48,48]

        hx1 = F.pixel_unshuffle(hxin,2) #[B,256,24,24]

        hx1 = self.rebnconv1(hx1) #[B,64,24,24]
        hx2 = F.pixel_unshuffle(hx1, 2)  # [B,256,12,12]
        hx2 = self.rebnconv2(hx2)  #[B,64,12,12]

        hx3 = self.rebnconv3_1(hx2)  #[B,64,12,12]

        B, C, h, w = hx3.shape
        fx_g = hx3.reshape(B, self.mid_ch, -1).transpose(1, 2) #[B,hw,C]
        x_g = self.global_attn(fx_g) #[B,hw,C]
        x_g = x_g.transpose(1, 2).reshape(B, self.mid_ch, h, w)
        hx3 = self.rebnconv3_2(x_g)


        hx2d = self.rebnconv2d(torch.cat((hx3,hx2),1)) #[B,256,12,12]
        hx2d = F.pixel_shuffle(hx2d, 2)  # [B,64,24,24]
        hx1d = self.rebnconv1d(torch.cat((hx2d,hx1),1)) # [B,256,24,24]
        hx1d = F.pixel_shuffle(hx1d, 2)  # [B,64,48,48]
        hxout = self.rebnconvout(hx1d) # [B,64,48,48]
        return hxout + hxin





class Nested_U(nn.Module):

    def __init__(self, in_ch=128, mid_ch=64,):
        super(Nested_U,self).__init__()
        self.rebnconvin  = nn.Sequential(nn.Conv2d(in_ch, 64, kernel_size=3, padding=1, bias=True),
                      nn.BatchNorm2d(64),
                      nn.LeakyReLU(inplace=True),
                                         )

        
        self.rebnconv1 = nn.Sequential(nn.Conv2d(256, 64, kernel_size=3, padding=1, bias=True),
                      nn.BatchNorm2d(64),
                      nn.LeakyReLU(inplace=True),
                                       )
        self.rebnconv2 = nn.Sequential(nn.Conv2d(256, 64, kernel_size=3, padding=1, bias=True),
                      nn.BatchNorm2d(64),
                      nn.LeakyReLU(inplace=True),
                                       )
        self.rebnconv3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True),
                      nn.BatchNorm2d(64),
                      nn.LeakyReLU(inplace=True),
                                       )

        self.rebnconv2d = nn.Sequential(nn.Conv2d(mid_ch*2,mid_ch*4, kernel_size=3, padding=1, bias=True),
                      nn.BatchNorm2d(mid_ch*4),
                      nn.LeakyReLU(inplace=True),
                      
                                        )

        self.rebnconv1d = nn.Sequential(nn.Conv2d(mid_ch*2,mid_ch*4, kernel_size=3, padding=1, bias=True),
                      nn.BatchNorm2d(mid_ch*4),
                      nn.LeakyReLU(inplace=True),
                     
                                        )

        self.rebnconvout  = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True),
                      nn.BatchNorm2d(64),
                      nn.LeakyReLU(inplace=True),
                                          )
        

    def forward(self,x):
        hxin = self.rebnconvin(x) # [B,64,48,48]

        hx1 = F.pixel_unshuffle(hxin,2) #[B,256,24,24]

        hx1 = self.rebnconv1(hx1) #[B,64,24,24]
        hx2 = F.pixel_unshuffle(hx1, 2)  # [B,256,12,12]
        hx2 = self.rebnconv2(hx2)  #[B,64,12,12]

        hx3 = self.rebnconv3(hx2)  #[B,64,12,12]

        hx2d = self.rebnconv2d(torch.cat((hx3,hx2),1)) #[B,256,12,12]
        hx2d = F.pixel_shuffle(hx2d, 2)  # [B,64,24,24]
        hx1d = self.rebnconv1d(torch.cat((hx2d,hx1),1)) # [B,256,24,24]
        hx1d = F.pixel_shuffle(hx1d, 2)  # [B,64,48,48]
        hxout = self.rebnconvout(hx1d) # [B,64,48,48]
        return hxout + hxin




