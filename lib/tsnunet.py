import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.optim import *
from lib.modules.NTU_NU import *
from lib.backbones.Res2Net_v1b import res2net50_v1b_26w_4s
from lib.backbones.SwinTransformer import SwinB
from lib.backbones.Encoder_pvt import Encoder_pvt_v2
from lib.backbones.MENet_resnet50 import MENet_ResNet
from lib.backbones.CvT import CvT_Backbone,CvT_21_384_IN22k


class TSNUNet_PVT_V2(nn.Module):
    def __init__(self, backbone, in_channels, depth=64, base_size=[384, 384], threshold=512, **kwargs):
        super(TSNUNet_PVT_V2, self).__init__()
        self.backbone = backbone
        self.in_channels = in_channels

        self.sod_loss_fn = lambda x, y: weighted_bce_loss_with_logits(x, y, reduction='mean') + iou_loss_with_logits(x,y,reduction='mean')

        self.conv_fuse345 = nn.Sequential(nn.Conv2d(192, 256, kernel_size=3, padding=1, bias=True),
                                          nn.BatchNorm2d(256),
                                          nn.LeakyReLU(inplace=True),
                                          nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=True),
                                          nn.BatchNorm2d(256),
                                          nn.LeakyReLU(inplace=True))

        self.conv_fuse = nn.Sequential(nn.Conv2d(15, 1, kernel_size=3, padding=1, bias=True),
                                       nn.BatchNorm2d(1),
                                       nn.LeakyReLU(inplace=True),
                                       nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=True),
                                       nn.BatchNorm2d(1),
                                       nn.LeakyReLU(inplace=True))

        self.conv1_LS = nn.Sequential(nn.Conv2d(4, 1, kernel_size=3, padding=1, bias=True),
                                       nn.BatchNorm2d(1),
                                       nn.LeakyReLU(inplace=True))
        self.conv2_LS = nn.Sequential(nn.Conv2d(4, 1, kernel_size=3, padding=1, bias=True),
                                      nn.BatchNorm2d(1),
                                      nn.LeakyReLU(inplace=True))
        self.conv345_LS = nn.Sequential(nn.Conv2d(4, 1, kernel_size=3, padding=1, bias=True),
                                       nn.BatchNorm2d(1),
                                       nn.LeakyReLU(inplace=True))



        self.conv5 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=True),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(80, 64, kernel_size=3, padding=1, bias=True),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=True),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(inplace=True))
        self.conv1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(inplace=True))
        self.conv345 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True),
                                     nn.BatchNorm2d(64),
                                     nn.LeakyReLU(inplace=True))
        self.unet45 = Nested_U(128, 64)
        self.unet34 = Nested_U(128, 64)

        self.unet2 = Nested_Trans_U(128, 64, 576) 
        self.unet1 = Nested_Trans_U(128, 64, 576)

        self.forward = self.forward_inference

    def to(self, device):
        super(TSNUNet_PVT_V2, self).to(device)
        return self

    def cuda(self, idx=None):
        if idx is None:
            idx = torch.cuda.current_device()
            print("idx", idx)

        self.to(device="cuda:{}".format(idx))
        return self

    def train(self, mode=True):
        super(TSNUNet_PVT_V2, self).train(mode)
        self.forward = self.forward_train
        return self

    def eval(self):
        super(TSNUNet_PVT_V2, self).train(False)
        self.forward = self.forward_inference
        return self

    def forward_inspyre(self, x):
        B, _, H, W = x.shape
        # print("x", x.shape)

        # x1, x2, x3, x4, x5 = self.backbone(x)

        x5, x4, x3, x2 = self.backbone(x)

        # print("x1", x1.shape)
        # print("x2", x2.shape)
        # print("x3", x3.shape)
        # print("x4", x4.shape)
        # print("x5", x5.shape)

        x2 = x2.transpose(1, 2).reshape(B, 64, 96, 96)
        x3 = x3.transpose(1, 2).reshape(B, 128, 48, 48)
        x4 = x4.transpose(1, 2).reshape(B, 320, 24, 24)
        x5 = x5.transpose(1, 2).reshape(B, 512, 12, 12)

        x5 = F.pixel_shuffle(x5, 4)  #  x5: [B, 32, 48, 48]
        x4 = F.pixel_shuffle(x4, 2)  #  x4: [B, 80, 48, 48]

        x5 = self.conv5(x5)  # x5: [B, 64, 48, 48]
        x4 = self.conv4(x4)  # x4: [B, 64, 48, 48]
        x3 = self.conv3(x3)  # x3: [B, 64, 48, 48]

        x45 = torch.cat([x4, x5], dim=1)  # [B, 192, 48, 48]
        x4 = self.unet45(x45)
        x34 = torch.cat([x3, x4], dim=1)  # [B, 192, 48, 48]
        x3 = self.unet34(x34)

        ef3, ef4, ef5 = x3, x4, x5  # [B, 64, 48, 48]


        x345 = torch.cat([ef3, ef4, ef5], dim=1)  # [B, 192, 48, 48]
        x345 = self.conv_fuse345(x345)  # [B, 256, 48, 48]
        x345 = F.pixel_shuffle(x345, 2)  # [B, 64, 96, 96]
        x345 = self.conv345(x345)

        x2 = self.conv2(x2)   
        x3_re = F.pixel_shuffle(x3, 2)  # [B, 16, 96, 96]
        x3_re = torch.cat([x3_re, x3_re, x3_re, x3_re], dim=1)  # [B, 64,96, 96]
        x3_re = self.conv1(x3_re)  #  x4: [B, 64, 48, 48]
        x1 = x3_re  # x4: [B, 64, 48, 48]

        x2_345 = torch.cat([x345, x2], dim=1)  # [B, 192, 48, 48]
        x2 = self.unet2(x2_345)
        x12 = torch.cat([x2, x1], dim=1)  # [B, 192, 48, 48]
        x1 = self.unet1(x12)
        ef1, ef2,ef345 = x1, x2, x345

        ef1 = F.pixel_shuffle(ef1, 4)   
        ef2 = F.pixel_shuffle(ef2, 4)   
        ef345 = F.pixel_shuffle(ef345, 4)   

        ef3 = F.pixel_shuffle(ef3, 8)   
        ef4 = F.pixel_shuffle(ef4, 8)   
        ef5 = F.pixel_shuffle(ef5, 8)   

        x0 = torch.cat([ef1, ef2,ef345,ef3, ef4, ef5], dim=1)
        x0 = self.conv_fuse(x0)
        ef1 = self.conv1_LS(ef1)
        ef2 = self.conv2_LS(ef2)
        ef345 = self.conv345_LS(ef345)

        return [x0,
                ef1,
                ef2,
               ef345,
                ]

    def forward_train(self, sample):
        x = sample['image']
        B, _, H, W = x.shape

        out = self.forward_inspyre(x)
        x0 = out[0]
        x1 = out[1]
        x2 = out[2]
        x3 = out[3]

        if type(sample) == dict and 'gt' in sample.keys() and sample['gt'] is not None:
            y = sample['gt']

            loss = self.sod_loss_fn(x3, y)
            loss += self.sod_loss_fn(x1, y)
            loss += self.sod_loss_fn(x2, y)
            loss += self.sod_loss_fn(x0, y)

        else:
            loss = 0

        pred = torch.sigmoid(x0)
        pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
        
        sample['pred'] = pred
        sample['loss'] = loss

        return sample

    def forward_inference(self, sample):
        B, _, H, W = sample['image'].shape

        out = self.forward_inspyre(sample['image'])
        x0 = out[0]
        pred = torch.sigmoid(x0)
        pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)

        sample['pred'] = pred
        sample['loss'] = 0
        return sample

def TSNUNet_PVT_V2_B2(depth, base_size, **kwargs):
    print("TSNUNet_PVT_V2")
    return TSNUNet_PVT_V2(Encoder_pvt_v2(), [64, 128, 320, 512],  depth, base_size, **kwargs)

class TSNUNet_res2net50(nn.Module):
    def __init__(self, backbone, in_channels, depth=64, base_size=[384, 384], threshold=512, **kwargs):
        super(TSNUNet_res2net50, self).__init__()
        self.backbone = backbone
        self.in_channels = in_channels

        self.sod_loss_fn = lambda x, y: weighted_bce_loss_with_logits(x, y, reduction='mean') + iou_loss_with_logits(x,y,reduction='mean')

        self.conv_fuse345 = nn.Sequential(nn.Conv2d(192, 256, kernel_size=3, padding=1, bias=True),
                                          nn.BatchNorm2d(256),
                                          nn.LeakyReLU(inplace=True),
                                          nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=True),
                                          nn.BatchNorm2d(256),
                                          nn.LeakyReLU(inplace=True))

        self.conv_fuse = nn.Sequential(nn.Conv2d(15, 1, kernel_size=3, padding=1, bias=True),
                                       nn.BatchNorm2d(1),
                                       nn.LeakyReLU(inplace=True),
                                       nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=True),
                                       nn.BatchNorm2d(1),
                                       nn.LeakyReLU(inplace=True))


        self.conv1_LS = nn.Sequential(nn.Conv2d(4, 1, kernel_size=3, padding=1, bias=True),
                                       nn.BatchNorm2d(1),
                                       nn.LeakyReLU(inplace=True))
        self.conv2_LS = nn.Sequential(nn.Conv2d(4, 1, kernel_size=3, padding=1, bias=True),
                                       nn.BatchNorm2d(1),
                                       nn.LeakyReLU(inplace=True))
        self.conv345_LS = nn.Sequential(nn.Conv2d(4, 1, kernel_size=3, padding=1, bias=True),
                                       nn.BatchNorm2d(1),
                                       nn.LeakyReLU(inplace=True))

        self.conv5 = nn.Sequential(nn.Conv2d(self.in_channels[4] // 16, 64, kernel_size=3, padding=1, bias=True),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(self.in_channels[3] // 4, 64, kernel_size=3, padding=1, bias=True),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(self.in_channels[2], 64, kernel_size=3, padding=1, bias=True),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(self.in_channels[1], 64, kernel_size=3, padding=1, bias=True),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(inplace=True))
        self.conv1 = nn.Sequential(nn.Conv2d(self.in_channels[0], 64, kernel_size=3, padding=1, bias=True),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(inplace=True))
        self.conv345 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True),
                                       nn.BatchNorm2d(64),
                                       nn.LeakyReLU(inplace=True))
        self.unet45 = Nested_U(128,64)
        self.unet34 = Nested_U(128,64)

        self.unet2 = Nested_Trans_U(128,64,576)
        self.unet1 = Nested_Trans_U(128,64,576)

        self.forward = self.forward_inference

    def to(self, device):
        super(TSNUNet_res2net50, self).to(device)
        return self

    def cuda(self, idx=None):
        if idx is None:
            idx = torch.cuda.current_device()
            print("idx", idx)

        self.to(device="cuda:{}".format(idx))
        return self

    def train(self, mode=True):
        super(TSNUNet_res2net50, self).train(mode)
        self.forward = self.forward_train
        return self

    def eval(self):
        super(TSNUNet_res2net50, self).train(False)
        self.forward = self.forward_inference
        return self

    def forward_inspyre(self, x):
        B, _, H, W = x.shape
        # print("x", x.shape)

        x1, x2, x3, x4, x5 = self.backbone(x)

        # print("x1", x1.shape)
        # print("x2", x2.shape)
        # print("x3", x3.shape)
        # print("x4", x4.shape)
        # print("x5", x5.shape)


        x5 = F.pixel_shuffle(x5, 4) 
        x4 = F.pixel_shuffle(x4, 2) 



        x5 = self.conv5(x5)
        x4 = self.conv4(x4)
        x3 = self.conv3(x3)

        x45 = torch.cat([x4, x5], dim=1)  
        x4 = self.unet45(x45)
        x34 = torch.cat([x3, x4], dim=1)  
        x3 = self.unet34(x34)

        ef3, ef4, ef5 = x3, x4, x5  

        x345 = torch.cat([ef3, ef4, ef5], dim=1)  
        x345 = self.conv_fuse345(x345)  
        x345 = F.pixel_shuffle(x345, 2)  
        x345 = self.conv345(x345)

        x2 = self.conv2(x2)  
        x1 = self.conv1(x1)  

        x2_345 = torch.cat([x345, x2], dim=1)  
        x2 = self.unet2(x2_345)
        x12 = torch.cat([x2, x1], dim=1)  
        x1 = self.unet1(x12)
        ef1, ef2, ef345 = x1, x2, x345

        ef1 = F.pixel_shuffle(ef1, 4)  
        ef2 = F.pixel_shuffle(ef2, 4)  
        ef345 = F.pixel_shuffle(ef345, 4)  

        ef3 = F.pixel_shuffle(ef3, 8)  
        ef4 = F.pixel_shuffle(ef4, 8)  
        ef5 = F.pixel_shuffle(ef5, 8)  

        x0 = torch.cat([ef1, ef2, ef345, ef3, ef4, ef5], dim=1)
        x0 = self.conv_fuse(x0)
        ef1 = self.conv1_LS(ef1)
        ef2 = self.conv2_LS(ef2)
        ef345 = self.conv345_LS(ef345)

        return [x0,
                ef1,
                ef2,
                ef345,
                ]

    def forward_train(self, sample):
        x = sample['image']
        B, _, H, W = x.shape
        
        out = self.forward_inspyre(x)
        x0 = out[0]
        x1 = out[1]
        x2 = out[2]
        x3 = out[3]

        if type(sample) == dict and 'gt' in sample.keys() and sample['gt'] is not None:
            y = sample['gt']


            loss = self.sod_loss_fn(x3, y)
            loss += self.sod_loss_fn(x1, y)
            loss += self.sod_loss_fn(x2, y)
            loss += self.sod_loss_fn(x0, y)

        else:
            loss = 0

        pred = torch.sigmoid(x0)
        pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)

        sample['pred'] = pred
        sample['loss'] = loss

        return sample

    def forward_inference(self, sample):
        B, _, H, W = sample['image'].shape


        out = self.forward_inspyre(sample['image'])
        x0 = out[0]

        pred = torch.sigmoid(x0)
        pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)

        sample['pred'] = pred
        sample['loss'] = 0
        return sample

def TSNUNet_res2net50_v1b_26w_4s(depth, pretrained, base_size, **kwargs):
    print("TSNUNet_res2net50")
    return TSNUNet_res2net50(res2net50_v1b_26w_4s(pretrained=pretrained), [64, 256, 512, 1024, 2048], depth,
                                     base_size, **kwargs)


class TSNUNet_resnet50(nn.Module):
    def __init__(self, backbone, in_channels, depth=64, base_size=[384, 384], threshold=512, **kwargs):
        super(TSNUNet_resnet50, self).__init__()
        self.backbone = backbone
        self.in_channels = in_channels


        self.sod_loss_fn = lambda x, y: weighted_bce_loss_with_logits(x, y, reduction='mean') + iou_loss_with_logits(x,
                                                                                                                     y,
                                                                                                                     reduction='mean')

        self.ret = lambda x, target: F.interpolate(x, size=target.shape[-2:], mode='bilinear', align_corners=False)
        self.res = lambda x, size: F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        self.des = lambda x, size: F.interpolate(x, size=size, mode='nearest')
        self.conv_fuse345 = nn.Sequential(nn.Conv2d(192, 256, kernel_size=3, padding=1, bias=True),
                                          nn.BatchNorm2d(256),
                                          nn.LeakyReLU(inplace=True),
                                          nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=True),
                                          nn.BatchNorm2d(256),
                                          nn.LeakyReLU(inplace=True))


        self.conv_fuse = nn.Sequential(nn.Conv2d(15, 1, kernel_size=3, padding=1, bias=True),
                                       nn.BatchNorm2d(1),
                                       nn.LeakyReLU(inplace=True),
                                       nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=True),
                                       nn.BatchNorm2d(1),
                                       nn.LeakyReLU(inplace=True))


        self.conv1_LS = nn.Sequential(nn.Conv2d(4, 1, kernel_size=3, padding=1, bias=True),
                                       nn.BatchNorm2d(1),
                                       nn.LeakyReLU(inplace=True))
        self.conv2_LS = nn.Sequential(nn.Conv2d(4, 1, kernel_size=3, padding=1, bias=True),
                                       nn.BatchNorm2d(1),
                                       nn.LeakyReLU(inplace=True))
        self.conv345_LS = nn.Sequential(nn.Conv2d(4, 1, kernel_size=3, padding=1, bias=True),
                                       nn.BatchNorm2d(1),
                                       nn.LeakyReLU(inplace=True))

        self.conv5 = nn.Sequential(nn.Conv2d(self.in_channels[4]//16, 64, kernel_size=3, padding=1, bias=True),
                                       nn.BatchNorm2d(64),
                                       nn.LeakyReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(self.in_channels[3]//4, 64, kernel_size=3, padding=1, bias=True),
                                       nn.BatchNorm2d(64),
                                       nn.LeakyReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(self.in_channels[2], 64, kernel_size=3, padding=1, bias=True),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(self.in_channels[1], 64, kernel_size=3, padding=1, bias=True),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(inplace=True))
        self.conv1 = nn.Sequential(nn.Conv2d(self.in_channels[0], 64, kernel_size=3, padding=1, bias=True),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(inplace=True))
        self.conv345 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True),
                                       nn.BatchNorm2d(64),
                                       nn.LeakyReLU(inplace=True))
        self.unet45 = Nested_U(128,64)
        self.unet34 = Nested_U(128,64)

        self.unet2 = Nested_Trans_U(128,64,576)
        self.unet1 = Nested_Trans_U(128,64,576)

        self.forward = self.forward_inference

    def to(self, device):  
        super(TSNUNet_resnet50, self).to(device)
        return self

    def cuda(self, idx=None):
        if idx is None:
            idx = torch.cuda.current_device()
            print("idx", idx)

        self.to(device="cuda:{}".format(idx))
        return self

    def train(self, mode=True):
        super(TSNUNet_resnet50, self).train(mode)
        self.forward = self.forward_train
        return self

    def eval(self):
        super(TSNUNet_resnet50, self).train(False)
        self.forward = self.forward_inference
        return self

    def forward_inspyre(self, x):
        B, _, H, W = x.shape
        # print("x", x.shape)

        x1, x2, x3, x4, x5 = self.backbone(x)

        # print("x1", x1.shape)
        # print("x2", x2.shape)
        # print("x3", x3.shape)
        # print("x4", x4.shape)
        # print("x5", x5.shape)


        x5 = F.pixel_shuffle(x5, 4) 
        x4 = F.pixel_shuffle(x4, 2)



        x5 = self.conv5(x5)
        x4 = self.conv4(x4)
        x3 = self.conv3(x3)

        x45 = torch.cat([x4, x5], dim=1)
        x4 = self.unet45(x45)
        x34 = torch.cat([x3, x4], dim=1)
        x3 = self.unet34(x34)
         
        ef3, ef4, ef5 = x3, x4, x5

        x345 = torch.cat([ef3, ef4, ef5], dim=1)
        x345 = self.conv_fuse345(x345)
        x345 = F.pixel_shuffle(x345, 2)
        x345 = self.conv345(x345)

        x2 = self.conv2(x2)   
        x1 = self.conv1(x1)

           

        x2_345 = torch.cat([x345, x2], dim=1)
        x2 = self.unet2(x2_345)
        x12 = torch.cat([x2, x1], dim=1)
        x1 = self.unet1(x12)
        ef1, ef2, ef345 = x1, x2, x345

        ef1 = F.pixel_shuffle(ef1, 4)
        ef2 = F.pixel_shuffle(ef2, 4)
        ef345 = F.pixel_shuffle(ef345, 4)

        ef3 = F.pixel_shuffle(ef3, 8)
        ef4 = F.pixel_shuffle(ef4, 8)
        ef5 = F.pixel_shuffle(ef5, 8)

        x0 = torch.cat([ef1, ef2, ef345, ef3, ef4, ef5], dim=1)
        x0 = self.conv_fuse(x0)
        ef1 = self.conv1_LS(ef1)
        ef2 = self.conv2_LS(ef2)
        ef345 = self.conv345_LS(ef345)

        return [x0,
                ef1,
                ef2,
                ef345,
                ]

    def forward_train(self, sample):
        x = sample['image']
        B, _, H, W = x.shape
        
        out = self.forward_inspyre(x)
        x0 = out[0]
        x1 = out[1]
        x2 = out[2]
        x3 = out[3]

        if type(sample) == dict and 'gt' in sample.keys() and sample['gt'] is not None:
            y = sample['gt']


            loss = self.sod_loss_fn(x3, y)
            loss += self.sod_loss_fn(x1, y)
            loss += self.sod_loss_fn(x2, y)
            loss += self.sod_loss_fn(x0, y)

        else:
            loss = 0

        pred = torch.sigmoid(x0)
        pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)

        sample['pred'] = pred
        sample['loss'] = loss
        
        
        return sample

    def forward_inference(self, sample):
        B, _, H, W = sample['image'].shape

        out = self.forward_inspyre(sample['image'])
        x0 = out[0]

        pred = torch.sigmoid(x0)
        pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)

        sample['pred'] = pred
        sample['loss'] = 0
        return sample

def TSNUNet_ResNet50_default(depth, pretrained, base_size, **kwargs):
    print("TSNUNet_ResNet50")
    return TSNUNet_resnet50(MENet_ResNet(), [64, 256, 512, 1024, 2048], depth, base_size, **kwargs)




class TSNUNet_SWinB(nn.Module):
    def __init__(self, backbone, in_channels, depth=64, base_size=[384, 384], threshold=512, **kwargs):
        super(TSNUNet_SWinB, self).__init__()
        self.backbone = backbone
        self.in_channels = in_channels


        self.sod_loss_fn = lambda x, y: weighted_bce_loss_with_logits(x, y, reduction='mean') + iou_loss_with_logits(x,y, reduction='mean')

        self.ret = lambda x, target: F.interpolate(x, size=target.shape[-2:], mode='bilinear', align_corners=False)
        self.res = lambda x, size: F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        self.des = lambda x, size: F.interpolate(x, size=size, mode='nearest')
        self.conv_fuse345 = nn.Sequential(nn.Conv2d(192, 256, kernel_size=3, padding=1, bias=True),
                                          nn.BatchNorm2d(256),
                                          nn.LeakyReLU(inplace=True),
                                          nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=True),
                                          nn.BatchNorm2d(256),
                                          nn.LeakyReLU(inplace=True))

        self.conv_fuse = nn.Sequential(nn.Conv2d(15, 1, kernel_size=3, padding=1, bias=True),
                                       nn.BatchNorm2d(1),
                                       nn.LeakyReLU(inplace=True),
                                       nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=True),
                                       nn.BatchNorm2d(1),
                                       nn.LeakyReLU(inplace=True))

        self.conv1_LS = nn.Sequential(nn.Conv2d(4, 1, kernel_size=3, padding=1, bias=True),
                                       nn.BatchNorm2d(1),
                                       nn.LeakyReLU(inplace=True))
        self.conv2_LS = nn.Sequential(nn.Conv2d(4, 1, kernel_size=3, padding=1, bias=True),
                                      nn.BatchNorm2d(1),
                                      nn.LeakyReLU(inplace=True))
        self.conv345_LS = nn.Sequential(nn.Conv2d(4, 1, kernel_size=3, padding=1, bias=True),
                                       nn.BatchNorm2d(1),
                                       nn.LeakyReLU(inplace=True))

        self.conv5 = nn.Sequential(nn.Conv2d(self.in_channels[4] // 16, 64, kernel_size=3, padding=1, bias=True),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(self.in_channels[3] // 4, 64, kernel_size=3, padding=1, bias=True),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(self.in_channels[2], 64, kernel_size=3, padding=1, bias=True),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(self.in_channels[1], 64, kernel_size=3, padding=1, bias=True),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(inplace=True))
        self.conv1 = nn.Sequential(nn.Conv2d(self.in_channels[0], 64, kernel_size=3, padding=1, bias=True),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(inplace=True))
        self.conv345 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True),
                                     nn.BatchNorm2d(64),
                                     nn.LeakyReLU(inplace=True))
        self.unet45 = Nested_U(128, 64)
        self.unet34 = Nested_U(128, 64)

        self.unet2 = Nested_Trans_U(128, 64, 576)
        self.unet1 = Nested_Trans_U(128, 64, 576)

        self.forward = self.forward_inference

    def to(self, device):
        super(TSNUNet_SWinB, self).to(device)
        return self

    def cuda(self, idx=None):
        if idx is None:
            idx = torch.cuda.current_device()
            print("idx", idx)

        self.to(device="cuda:{}".format(idx))
        return self

    def train(self, mode=True):
        super(TSNUNet_SWinB, self).train(mode)
        self.forward = self.forward_train
        return self

    def eval(self):
        super(TSNUNet_SWinB, self).train(False)
        self.forward = self.forward_inference
        return self

    def forward_inspyre(self, x):
        B, _, H, W = x.shape
        # print("x", x.shape)

        x1, x2, x3, x4, x5 = self.backbone(x)

        # print("x1", x1.shape)
        # print("x2", x2.shape)
        # print("x3", x3.shape)
        # print("x4", x4.shape)
        # print("x5", x5.shape)

        x5 = F.pixel_shuffle(x5, 4)   
        x4 = F.pixel_shuffle(x4, 2)

        x5 = self.conv5(x5)
        x4 = self.conv4(x4)
        x3 = self.conv3(x3)

        x45 = torch.cat([x4, x5], dim=1)
        x4 = self.unet45(x45)
        x34 = torch.cat([x3, x4], dim=1)
        x3 = self.unet34(x34)

         
        ef3, ef4, ef5 = x3, x4, x5

          
        x345 = torch.cat([ef3, ef4, ef5], dim=1)
        x345 = self.conv_fuse345(x345)
        x345 = F.pixel_shuffle(x345, 2)
        x345 = self.conv345(x345)

        x2 = self.conv2(x2)   
        x1 = self.conv1(x1)

           

        x2_345 = torch.cat([x345, x2], dim=1)
        x2 = self.unet2(x2_345)
        x12 = torch.cat([x2, x1], dim=1)
        x1 = self.unet1(x12)
        ef1, ef2, ef345 = x1, x2, x345

        ef1 = F.pixel_shuffle(ef1, 4)   
        ef2 = F.pixel_shuffle(ef2, 4)   
        ef345 = F.pixel_shuffle(ef345, 4)   

        ef3 = F.pixel_shuffle(ef3, 8)   
        ef4 = F.pixel_shuffle(ef4, 8)   
        ef5 = F.pixel_shuffle(ef5, 8)   

        x0 = torch.cat([ef1, ef2, ef345, ef3, ef4, ef5], dim=1)
        x0 = self.conv_fuse(x0)
        ef1 = self.conv1_LS(ef1)
        ef2 = self.conv2_LS(ef2)
        ef345 = self.conv345_LS(ef345)

        return [x0,
                ef1,
                ef2,
                ef345
                ]

    def forward_train(self, sample):
        x = sample['image']
        B, _, H, W = x.shape
        
        out = self.forward_inspyre(x)
        x0 = out[0]
        x1 = out[1]
        x2 = out[2]
        x3 = out[3]

        if type(sample) == dict and 'gt' in sample.keys() and sample['gt'] is not None:
            y = sample['gt']

            loss = self.sod_loss_fn(x3, y)
            loss += self.sod_loss_fn(x1, y)
            loss += self.sod_loss_fn(x2, y)
            loss += self.sod_loss_fn(x0, y)

        else:
            loss = 0

        pred = torch.sigmoid(x0)
        pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)

        sample['pred'] = pred
        sample['loss'] = loss

        return sample

    def forward_inference(self, sample):
        B, _, H, W = sample['image'].shape

        out = self.forward_inspyre(sample['image'])
        x0 = out[0]

        pred = torch.sigmoid(x0)
        pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)

        sample['pred'] = pred
        sample['loss'] = 0
        return sample

def TSNUNet_SWinB_patch4_window12_384_22kto1k(depth, pretrained, base_size, **kwargs):
    print("TSNUNet_SWinB")
    return TSNUNet_SWinB(SwinB(pretrained=pretrained),[128, 128, 256, 512, 1024], depth, base_size,**kwargs)

class TSNUNet_CvT(nn.Module):
    """
    TSNUNet with CvT backbone
    """
    def __init__(self, backbone=None, in_channels=3, depth=64, base_size=[384, 384], threshold=512, **kwargs):
        super(TSNUNet_CvT, self).__init__()

        if backbone is None:
            backbone = CvT_Backbone(pretrained=True)

        self.backbone = backbone
        self.in_channels = in_channels
        self.depth = depth
        self.base_size = base_size
        self.threshold = threshold

        self.sod_loss_fn = lambda x, y: weighted_bce_loss_with_logits(x, y, reduction='mean') + iou_loss_with_logits(x,
                                                                                                                     y,
                                                                                                                     reduction='mean')

        self.ret = lambda x, target: F.interpolate(x, size=target.shape[-2:], mode='bilinear', align_corners=False)
        self.res = lambda x, size: F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        self.des = lambda x, size: F.interpolate(x, size=size, mode='nearest')
        self.conv_fuse345 = nn.Sequential(nn.Conv2d(192, 256, kernel_size=3, padding=1, bias=True),
                                          nn.BatchNorm2d(256),
                                          nn.LeakyReLU(inplace=True),
                                          nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=True),
                                          nn.BatchNorm2d(256),
                                          nn.LeakyReLU(inplace=True))

        self.conv_fuse = nn.Sequential(nn.Conv2d(15, 1, kernel_size=3, padding=1, bias=True),
                                       nn.BatchNorm2d(1),
                                       nn.LeakyReLU(inplace=True),
                                       nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=True),
                                       nn.BatchNorm2d(1),
                                       nn.LeakyReLU(inplace=True))

        self.conv1_LS = nn.Sequential(nn.Conv2d(4, 1, kernel_size=3, padding=1, bias=True),
                                       nn.BatchNorm2d(1),
                                       nn.LeakyReLU(inplace=True))
        self.conv2_LS = nn.Sequential(nn.Conv2d(4, 1, kernel_size=3, padding=1, bias=True),
                                      nn.BatchNorm2d(1),
                                      nn.LeakyReLU(inplace=True))
        self.conv345_LS = nn.Sequential(nn.Conv2d(4, 1, kernel_size=3, padding=1, bias=True),
                                       nn.BatchNorm2d(1),
                                       nn.LeakyReLU(inplace=True))

        self.conv5 = nn.Sequential(nn.Conv2d(96, 64, kernel_size=3, padding=1, bias=True),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(96, 64, kernel_size=3, padding=1, bias=True),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(192, 64, kernel_size=3, padding=1, bias=True),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(inplace=True))
        self.conv1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(inplace=True))
        self.conv345 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True),
                                     nn.BatchNorm2d(64),
                                     nn.LeakyReLU(inplace=True))
        self.unet45 = Nested_U(128, 64)
        self.unet34 = Nested_U(128, 64)

        self.unet2 = Nested_Trans_U(128, 64, 576)
        self.unet1 = Nested_Trans_U(128, 64, 576)

        self.forward = self.forward_inference

    def to(self, device):
        super(TSNUNet_CvT, self).to(device)
        return self

    def cuda(self, idx=None):
        if idx is None:
            idx = torch.cuda.current_device()
        self.to(device="cuda:{}".format(idx))
        return self

    def forward_inspyre(self, x):
        B, _, H, W = x.shape


        x1, x2, x3, x4, x5 = self.backbone(x)

        # print("x1", x1.shape)
        # print("x2", x2.shape)
        # print("x3", x3.shape)
        # print("x4", x4.shape)
        # print("x5", x5.shape)

        x5 = F.pixel_shuffle(x5, 2)   
        x4 = F.pixel_shuffle(x4, 2)

        x5 = self.conv5(x5)
        x4 = self.conv4(x4)
        x3 = self.conv3(x3)

        x45 = torch.cat([x4, x5], dim=1)
        x4 = self.unet45(x45)
        x34 = torch.cat([x3, x4], dim=1)
        x3 = self.unet34(x34)

        ef3, ef4, ef5 = x3, x4, x5

        x345 = torch.cat([ef3, ef4, ef5], dim=1)
        x345 = self.conv_fuse345(x345)
        x345 = F.pixel_shuffle(x345, 2)
        x345 = self.conv345(x345)

        x2 = self.conv2(x2)
        x3_re = F.pixel_shuffle(x3, 2)
        x3_re = torch.cat([x3_re, x3_re, x3_re, x3_re], dim=1)
        x3_re = self.conv1(x3_re)
        x1 = x3_re

        x2_345 = torch.cat([x345, x2], dim=1)
        x2 = self.unet2(x2_345)
        x12 = torch.cat([x2, x1], dim=1)
        x1 = self.unet1(x12)
        ef1, ef2, ef345 = x1, x2, x345

        ef1 = F.pixel_shuffle(ef1, 4)   
        ef2 = F.pixel_shuffle(ef2, 4)   
        ef345 = F.pixel_shuffle(ef345, 4)   

        ef3 = F.pixel_shuffle(ef3, 8)   
        ef4 = F.pixel_shuffle(ef4, 8)   
        ef5 = F.pixel_shuffle(ef5, 8)   

        x0 = torch.cat([ef1, ef2, ef345, ef3, ef4, ef5], dim=1)
        x0 = self.conv_fuse(x0)
        ef1 = self.conv1_LS(ef1)
        ef2 = self.conv2_LS(ef2)
        ef345 = self.conv345_LS(ef345)

        return [x0,
                ef1,
                ef2,
                ef345,
                ]

    def forward_train(self, sample):
        x = sample['image']
        B, _, H, W = x.shape
        out = self.forward_inspyre(x)
        x0 = out[0]
        x1 = out[1]
        x2 = out[2]
        x3 = out[3]

        if type(sample) == dict and 'gt' in sample.keys() and sample['gt'] is not None:
            y = sample['gt']

            loss = self.sod_loss_fn(x3, y)
            loss += self.sod_loss_fn(x1, y)
            loss += self.sod_loss_fn(x2, y)
            loss += self.sod_loss_fn(x0, y)

        else:
            loss = 0

        pred = torch.sigmoid(x0)
        pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)

        sample['pred'] = pred
        sample['loss'] = loss
        return sample

    def forward_inference(self, sample):
        B, _, H, W = sample['image'].shape

        out = self.forward_inspyre(sample['image'])
        x0 = out[0]
        pred = torch.sigmoid(x0)
        pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)

        sample['pred'] = pred
        sample['loss'] = 0
        return sample
    def train(self, mode=True):
        super(TSNUNet_CvT, self).train(mode)
        self.forward = self.forward_train
        return self

    def eval(self):
        super(TSNUNet_CvT, self).train(False)
        self.forward = self.forward_inference
        return self

def TSNUNet_CvT_21_384_IN22k(depth, pretrained, base_size, **kwargs):
    print("TSNUNet_CvT")
    return TSNUNet_CvT(CvT_21_384_IN22k(pretrained=pretrained),[128, 128, 256, 512, 1024], depth, base_size,**kwargs)



class TSNUNet_FusedBackbones(nn.Module):
    def __init__(self, depth=64, base_size=[384, 384], threshold=512, pretrained=True, **kwargs):
        super(TSNUNet_FusedBackbones, self).__init__()

        self.pvt_backbone = Encoder_pvt_v2()
        self.res2net_backbone = res2net50_v1b_26w_4s(pretrained=pretrained)

        # Define in_channels for each backbone
        self.pvt_in_channels = [64, 128, 320, 512]  # PVT_V2 outputs (x2, x3, x4, x5)
        self.res2net_in_channels = [64, 256, 512, 1024, 2048]  # Res2net50 outputs (x1, x2, x3, x4, x5)

        # Fusion strategy: PVT x2 + Res2net x1 and x2, PVT x3 + Res2net x3, PVT x4 + Res2net x4, PVT x5 + Res2net x5
        self.fused_in_channels_stage1 = self.pvt_in_channels[0] + self.res2net_in_channels[
            0]  # For x2 of PVT + x1 of Res2net
        self.fused_in_channels_stage2 = self.pvt_in_channels[0] + self.res2net_in_channels[
            1]  # For x2 of PVT + x2 of Res2net
        self.fused_in_channels_stage3 = self.pvt_in_channels[1] + self.res2net_in_channels[
            2]  # For x3 of PVT + x3 of Res2net
        self.fused_in_channels_stage4 = self.pvt_in_channels[2] + self.res2net_in_channels[
            3]  # For x4 of PVT + x4 of Res2net
        self.fused_in_channels_stage5 = self.pvt_in_channels[3] + self.res2net_in_channels[
            4]  # For x5 of PVT + x5 of Res2net

        self.depth = depth
        self.base_size = base_size
        self.threshold = threshold

        self.sod_loss_fn = lambda x, y: weighted_bce_loss_with_logits(x, y, reduction='mean') + iou_loss_with_logits(x,y,reduction='mean')

        self.ret = lambda x, target: F.interpolate(x, size=target.shape[-2:], mode='bilinear', align_corners=False)
        self.res = lambda x, size: F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        self.des = lambda x, size: F.interpolate(x, size=size, mode='nearest')

        # Adjust conv_fuse345 for combined channels
        self.conv_fuse345 = nn.Sequential(nn.Conv2d(192, 256, kernel_size=3, padding=1, bias=True),
                                          nn.BatchNorm2d(256),
                                          nn.LeakyReLU(inplace=True),
                                          nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=True),
                                          nn.BatchNorm2d(256),
                                          nn.LeakyReLU(inplace=True))

        self.conv_fuse = nn.Sequential(nn.Conv2d(15, 1, kernel_size=3, padding=1, bias=True),
                                       nn.BatchNorm2d(1),
                                       nn.LeakyReLU(inplace=True),
                                       nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=True),
                                       nn.BatchNorm2d(1),
                                       nn.LeakyReLU(inplace=True))

        self.conv1_LS = nn.Sequential(nn.Conv2d(4, 1, kernel_size=3, padding=1, bias=True),
                                       nn.BatchNorm2d(1),
                                       nn.LeakyReLU(inplace=True))
        self.conv2_LS = nn.Sequential(nn.Conv2d(4, 1, kernel_size=3, padding=1, bias=True),
                                      nn.BatchNorm2d(1),
                                      nn.LeakyReLU(inplace=True))
        self.conv345_LS = nn.Sequential(nn.Conv2d(4, 1, kernel_size=3, padding=1, bias=True),
                                       nn.BatchNorm2d(1),
                                       nn.LeakyReLU(inplace=True))

        # Adjust conv layers to handle combined features. This needs careful consideration.
        self.conv5 = nn.Sequential(
            nn.Conv2d(self.fused_in_channels_stage5 // 16, 64, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(self.fused_in_channels_stage4 // 4, 64, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(self.fused_in_channels_stage3, 64, kernel_size=3, padding=1, bias=True),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(self.fused_in_channels_stage2, 64, kernel_size=3, padding=1, bias=True),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(inplace=True))
        self.conv1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(inplace=True))

        self.conv345 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True),
                                     nn.BatchNorm2d(64),
                                     nn.LeakyReLU(inplace=True))
        self.unet45 = Nested_U(128, 64)
        self.unet34 = Nested_U(128, 64)

        self.unet2 = Nested_Trans_U(128, 64, 576)
        self.unet1 = Nested_Trans_U(128, 64, 576)

        self.forward = self.forward_inference

    def to(self, device):
        super(TSNUNet_FusedBackbones, self).to(device)
        return self

    def cuda(self, idx=None):
        if idx is None:
            idx = torch.cuda.current_device()
        self.to(device="cuda:{}".format(idx))
        return self

    def train(self, mode=True):
        super(TSNUNet_FusedBackbones, self).train(mode)
        self.forward = self.forward_train
        return self

    def eval(self):
        super(TSNUNet_FusedBackbones, self).train(False)
        self.forward = self.forward_inference
        return self

    def forward_inspyre(self, x):
        B, _, H, W = x.shape

        # Extract features from PVT_V2
        pvt_x5, pvt_x4, pvt_x3, pvt_x2 = self.pvt_backbone(x)

        # Adapt PVT_V2 outputs to match expected shapes
        pvt_x2 = pvt_x2.transpose(1, 2).reshape(B, self.pvt_in_channels[0], H // 4, W // 4)  # 64
        pvt_x3 = pvt_x3.transpose(1, 2).reshape(B, self.pvt_in_channels[1], H // 8, W // 8)  # 128
        pvt_x4 = pvt_x4.transpose(1, 2).reshape(B, self.pvt_in_channels[2], H // 16, W // 16)  # 320
        pvt_x5 = pvt_x5.transpose(1, 2).reshape(B, self.pvt_in_channels[3], H // 32, W // 32)  # 512

        # Extract features from Res2net50
        res2net_x1, res2net_x2, res2net_x3, res2net_x4, res2net_x5 = self.res2net_backbone(x)

        # --- Feature Fusion Strategy ---
        fused_x2 = torch.cat([pvt_x2, res2net_x2], dim=1)  # 64 + 256 = 320 channels

        # Stage 2 (H/8 resolution): Combine pvt_x3 and res2net_x3
        fused_x3 = torch.cat([pvt_x3, res2net_x3], dim=1)  # 128 + 512 = 640 channels

        # Stage 3 (H/16 resolution): Combine pvt_x4 and res2net_x4
        fused_x4 = torch.cat([pvt_x4, res2net_x4], dim=1)  # 320 + 1024 = 1344 channels

        # Stage 4 (H/32 resolution): Combine pvt_x5 and res2net_x5
        fused_x5 = torch.cat([pvt_x5, res2net_x5], dim=1)  # 512 + 2048 = 2560 channels


        fused_x5_processed = F.pixel_shuffle(fused_x5, 4)  # [B, 2560//16, H/8, W/8] = [B, 160, H/8, W/8]
        fused_x4_processed = F.pixel_shuffle(fused_x4, 2)  # [B, 1344//4, H/8, W/8] = [B, 336, H/8, W/8]

        x5 = self.conv5(fused_x5_processed)  # Assuming output is 64 channels
        x4 = self.conv4(fused_x4_processed)  # Assuming output is 64 channels
        x3 = self.conv3(fused_x3)  # Assuming output is 64 channels

        x45 = torch.cat([x4, x5], dim=1)  # [B, 128, H/8, W/8]
        x4 = self.unet45(x45)
        x34 = torch.cat([x3, x4], dim=1)  # [B, 128, H/8, W/8]
        x3 = self.unet34(x34)

        ef3, ef4, ef5 = x3, x4, x5

        x345 = torch.cat([ef3, ef4, ef5], dim=1)
        x345 = self.conv_fuse345(x345)
        x345 = F.pixel_shuffle(x345, 2)
        x345 = self.conv345(x345)

        x2 = self.conv2(fused_x2)  # Assuming output is 64 channels
        # x1 from PVT_V2 is essentially pvt_x2 processed. I'll need to re-evaluate this for consistency.
        # For now, let's process the fused_x2 and a processed version of fused_x3 for x1.
        x3_re = F.pixel_shuffle(x3, 2)
        x3_re = torch.cat([x3_re, x3_re, x3_re, x3_re], dim=1)  # [B, 64, H/4, W/4]
        x3_re = self.conv1(x3_re)  # Assuming output is 64 channels
        x1 = x3_re

        x2_345 = torch.cat([x345, x2], dim=1)
        x2 = self.unet2(x2_345)
        x12 = torch.cat([x2, x1], dim=1)
        x1 = self.unet1(x12)
        ef1, ef2, ef345 = x1, x2, x345

        ef1 = F.pixel_shuffle(ef1, 4)
        ef2 = F.pixel_shuffle(ef2, 4)
        ef345 = F.pixel_shuffle(ef345, 4)

        ef3 = F.pixel_shuffle(ef3, 8)
        ef4 = F.pixel_shuffle(ef4, 8)
        ef5 = F.pixel_shuffle(ef5, 8)

        x0 = torch.cat([ef1, ef2, ef345, ef3, ef4, ef5], dim=1)
        x0 = self.conv_fuse(x0)
        ef1 = self.conv1_LS(ef1)
        ef2 = self.conv2_LS(ef2)
        ef345 = self.conv345_LS(ef345)

        return [x0, ef1, ef2, ef345]

    def forward_train(self, sample):
        x = sample['image']
        out = self.forward_inspyre(x)
        x0, x1, x2, x3 = out[0], out[1], out[2], out[3]

        if type(sample) == dict and 'gt' in sample.keys() and sample['gt'] is not None:
            y = sample['gt']
            loss = self.sod_loss_fn(x3, y)
            loss += self.sod_loss_fn(x1, y)
            loss += self.sod_loss_fn(x2, y)
            loss += self.sod_loss_fn(x0, y)
        else:
            loss = 0

        pred = torch.sigmoid(x0)
        pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
        sample['pred'] = pred
        sample['loss'] = loss
        return sample

    def forward_inference(self, sample):
        B, _, H, W = sample['image'].shape

        out = self.forward_inspyre(sample['image'])
        x0 = out[0]


        pred = torch.sigmoid(x0)
        pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)

        sample['pred'] = pred
        sample['loss'] = 0
        return sample


def TSNUNet_FusedBackbones_Factory(depth, base_size, pretrained=True, **kwargs):
    print(f"Creating TSNUNet Fused PVT_V2 and Res2net50 backbones")
    return TSNUNet_FusedBackbones(depth=depth, base_size=base_size, pretrained=pretrained, **kwargs)



