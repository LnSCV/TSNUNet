import torch
import torch.nn as nn
import torch.nn.functional as F

class Nested_U(nn.Module):
    def __init__(self, in_ch=128, mid_ch=64, ):
        super(Nested_U, self).__init__()
        self.rebnconvin = nn.Sequential(nn.Conv2d(in_ch, 64, kernel_size=3, padding=1, bias=True),
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

        self.rebnconv2d = nn.Sequential(nn.Conv2d(mid_ch * 2, mid_ch * 4, kernel_size=3, padding=1, bias=True),
                                        nn.BatchNorm2d(mid_ch * 4),
                                        nn.LeakyReLU(inplace=True),

                                        )

        self.rebnconv1d = nn.Sequential(nn.Conv2d(mid_ch * 2, mid_ch * 4, kernel_size=3, padding=1, bias=True),
                                        nn.BatchNorm2d(mid_ch * 4),
                                        nn.LeakyReLU(inplace=True),

                                        )

        self.rebnconvout = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True),
                                         nn.BatchNorm2d(64),
                                         nn.LeakyReLU(inplace=True),
                                         )

    def forward(self, x):
        hxin = self.rebnconvin(x)  # [B,64,48,48]

        hx1 = F.pixel_unshuffle(hxin, 2)  # [B,256,24,24]

        hx1 = self.rebnconv1(hx1)  # [B,64,24,24]
        hx2 = F.pixel_unshuffle(hx1, 2)  # [B,256,12,12]
        hx2 = self.rebnconv2(hx2)  # [B,64,12,12]

        hx3 = self.rebnconv3(hx2)  # [B,64,12,12]

        hx2d = self.rebnconv2d(torch.cat((hx3, hx2), 1))  # [B,256,12,12]
        hx2d = F.pixel_shuffle(hx2d, 2)  # [B,64,24,24]
        hx1d = self.rebnconv1d(torch.cat((hx2d, hx1), 1))  # [B,256,24,24]
        hx1d = F.pixel_shuffle(hx1d, 2)  # [B,64,48,48]
        hxout = self.rebnconvout(hx1d)  # [B,64,48,48]
        return hxout + hxin


# Instantiate the model
model = Nested_U(in_ch=128, mid_ch=64)

# Calculate the number of parameters
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters in Nested_U: {total_params / 1e6:.3f} M")