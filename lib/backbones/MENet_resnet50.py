import timm
import torch.nn as nn
import torch.nn.functional as F
class MENet_ResNet(nn.Module):
    def __init__(self):
        super(MENet_ResNet, self).__init__()
        self.resnet = timm.create_model(model_name="resnet50", pretrained=True, in_chans=3, features_only=True)
        #self.resnet = timm.create_model(model_name="resnet101", pretrained=True, in_chans=3, features_only=True)
        self.initialize()
        self.conv1 = nn.Sequential(nn.Conv2d(256, 64, kernel_size=3, padding=1, bias=True),
                                       nn.BatchNorm2d(64),
                                       nn.LeakyReLU(inplace=True),
                                       nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True),
                                       nn.BatchNorm2d(64),
                                       nn.LeakyReLU(inplace=True))

    def forward(self, x):
        out1, out2, out3, out4, out5 = self.resnet(x)
        out1 =  F.pixel_unshuffle(out1, 2)
        out1 = self.conv1(out1)

        return out1, out2, out3, out4, out5

    def initialize(self):
        pass


class MENet_ResNet101(nn.Module):
    def __init__(self):
        super(MENet_ResNet101, self).__init__()
        # self.resnet = timm.create_model(model_name="resnet50", pretrained=True, in_chans=3, features_only=True)
        self.resnet = timm.create_model(model_name="resnet101", pretrained=True, in_chans=3, features_only=True)
        self.initialize()
        self.conv1 = nn.Sequential(nn.Conv2d(256, 64, kernel_size=3, padding=1, bias=True),
                                       nn.BatchNorm2d(64),
                                       nn.LeakyReLU(inplace=True),
                                       nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True),
                                       nn.BatchNorm2d(64),
                                       nn.LeakyReLU(inplace=True))

    def forward(self, x):
        out1, out2, out3, out4, out5 = self.resnet(x)
        out1 =  F.pixel_unshuffle(out1, 2)
        out1 = self.conv1(out1)

        return out1, out2, out3, out4, out5

    def initialize(self):
        pass