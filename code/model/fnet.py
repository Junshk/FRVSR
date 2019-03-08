import torch
import torch.nn as nn
from model import common
class FNet(nn.Module):
    def __init__(self, args):
        super(FNet, self).__init__()
        # global average pooling: feature --> point
        # feature channel downscale and upscale --> channel weight
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)
        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)
        
        self.fnet = nn.Sequential(
                nn.Conv2d(6, 32, 3, 1, 1),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(32, 32, 3, 1, 1),
                nn.LeakyReLU(0.2, True),
                nn.MaxPool2d(2),

                nn.Conv2d(32, 64, 3, 1, 1),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(64, 64, 3, 1, 1),
                nn.LeakyReLU(0.2, True),
                nn.MaxPool2d(2),

                nn.Conv2d(64, 128, 3, 1, 1),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(128, 128, 3, 1, 1),
                nn.LeakyReLU(0.2, True),
                nn.MaxPool2d(2),

                nn.Conv2d(128, 256, 3, 1, 1),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(256, 256, 3, 1, 1),
                nn.LeakyReLU(0.2, True),
                nn.Upsample(scale_factor=2, mode='bilinear'),

                nn.Conv2d(256, 128, 3, 1, 1),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(128, 128, 3, 1, 1),
                nn.LeakyReLU(0.2, True),
                nn.Upsample(scale_factor=2, mode='bilinear'),
                
                nn.Conv2d(128, 64, 3, 1, 1),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(64, 64, 3, 1, 1),
                nn.LeakyReLU(0.2, True),
                nn.Upsample(scale_factor=2, mode='bilinear'),
                
                
                nn.Conv2d(64, 32, 3, 1, 1),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(32, 2, 3, 1, 1),
                
                nn.Tanh(),
                )
    

    def forward(self, x, x_):
        x = self.sub_mean(x)
        x_ = self.sub_mean(x_)
        x = self.fnet(torch.cat([x, x_], 1))
        return x#self.add_mean(x)
