import torch
import torch.nn as nn
from torch.nn import functional as F
from timm.models.layers import trunc_normal_, DropPath
from torch.serialization import add_safe_globals
from thop import profile
import os

from models.DDAttention import *
from models.CrossSimilarityAttention import *
from models import MixedFeatureNet



class QDDANet(nn.Module):
    def __init__(self, num_class=7, num_head=2, embed_dim=786,pretrained=True):
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
        self.num_head = num_head
        for i in range(int(num_head)):
            setattr(self, "cat_head%d" % (i), CoordAttHead())

        self.cls_head = ClassifierHead(num_class=num_class)

        self.cross_similarity_attention = CrossAttention(embed_dim=embed_dim)


    def forward(self, x_anchor, x_positive, x_negative, x_negative2):
        """
        :param x_anchor:
        :param x_positive:
        :param x_negative:
        :param x_negative2:
        :return: cls_bases and cls_cross
        """

        '''--------------------------anchor------------------------------------'''
        x_anchor = self.backbone(x_anchor)  # using MNF to extract features.
        # Tensor.shape(b, 7, 7, 512)
        cls_base_anchor, x_at_anchor = self.cls_head(x_anchor)
        # if x_positive is None:  # infera
        #     return x_at_anchor
        '''--------------------------------------------------------------------'''

        '''---------------------------positive-----------------------------------'''
        x_positive = self.backbone(x_positive)
        cls_positive, x_at_positive = self.cls_head(x_positive)
        '''----------------------------------------------------------------------'''

        '''---------------------------negative-----------------------------------'''
        x_negative = self.backbone(x_negative)
        cls_negative, x_at_negative = self.cls_head(x_negative)
        '''----------------------------------------------------------------------'''

        '''---------------------------negative2----------------------------------'''
        x_negative2 = self.backbone(x_negative2)
        cls_negative2, x_at_negative2 = self.cls_head(x_negative2)
        '''----------------------------------------------------------------------'''

        x_csa_anchor, x_csa_positive, x_csa_negative, x_csa_negative2, _ = self.cross_similarity_attention(x_at_anchor,
                                                                                    x_positive, x_negative, x_negative2)

        x_at_anchor = x_at_anchor + x_csa_anchor
        x_at_positive = x_at_positive + x_csa_positive
        x_negative = x_negative + x_csa_negative
        x_negative2 = x_negative2 + x_csa_negative2

        x_at_anchor,_ = self.cls_head(x_at_anchor)
        x_at_positive,_ = self.cls_head(x_at_positive)
        x_negative,_ = self.cls_head(x_negative)
        x_negative2,_ = self.cls_head(x_negative2)

        return cls_base_anchor, cls_positive, cls_negative, cls_negative2, x_at_anchor, x_at_positive, x_negative, x_negative2