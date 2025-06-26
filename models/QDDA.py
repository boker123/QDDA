import torch
import torch.nn as nn
from torch.nn import functional as F
from timm.models.layers import trunc_normal_, DropPath
from torch.serialization import add_safe_globals
from thop import profile
from models import *
import os

from models.DDAttention import *
from models.MixedFeatureNet import *

class Linear_block(nn.Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Linear_block, self).__init__()
        self.conv = nn.Conv2d(in_c, out_channels=out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class QDDANet(nn.Module):
    def __init__(self, pretrained):
        super(QDDANet, self).__init__()

        mixed_feature = MixedFeatureNet.MixedFeatureNet()
        if pretrained:
            # Check if CUDA is available and set a device accordingly
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            # First, try the standard approach with proper safe globals
            try:
                # Add all necessary classes to safe globals
                add_safe_globals({
                    'MixedFeatureNet': MixedFeatureNet.MixedFeatureNet,
                    'Linear_block': Linear_block,
                    'Flatten': Flatten,
                    'CoordAttHead': CoordAttHead,
                    'CoordAtt': CoordAtt,
                    'h_sigmoid': h_sigmoid,
                    'h_swish': h_swish
                })
                mixed_feature = torch.load(os.path.join('./pretrained/', "MFN_msceleb.pth"))
            except:
                # Fallback: Disable safe loading entirely
                mixed_feature = torch.load(os.path.join('./pretrained/', "MFN_msceleb.pth"),
                                 weights_only=False, map_location=device)
      
        self.backbone = nn.Sequential(*list(mixed_feature.children())[:-4])
