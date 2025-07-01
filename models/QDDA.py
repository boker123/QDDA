from torch.serialization import add_safe_globals
import os

from models.DDAttention import *
from models.CrossSimilarityAttention import *
from networks import MixedFeatureNet


class PositionalEmbedding(nn.Module):
    def __init__(self, input_channels, embed_dim, H, W, use_projection=True):
        """
        Args:
            input_channels: 输入特征的通道数，如512
            embed_dim: 输出嵌入维度
            H, W: 输入特征图的空间尺寸（如7x7）
            use_projection: 是否加入1x1卷积投影通道
        """
        super(PositionalEmbedding, self).__init__()
        self.H = H
        self.W = W
        self.N = H * W
        self.embed_dim = embed_dim

        # 可选：通过 1x1 conv 将 C → embed_dim（如果 C != embed_dim）
        self.proj = nn.Conv2d(input_channels, embed_dim, kernel_size=1) if use_projection else nn.Identity()

        # 位置编码，大小为 [1, N, embed_dim]
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.N, embed_dim))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)  # 初始化为正态分布

    def forward(self, x):
        """
        Args:
            x: 输入张量 [B, C, H, W]
        Returns:
            带位置编码的序列表示 [B, N, embed_dim]
        """
        B = x.size(0)
        x = self.proj(x)              # [B, C, H, W] → [B, embed_dim, H, W]
        x = x.view(B, self.embed_dim, self.N)  # → [B, embed_dim, N]
        x = x.permute(0, 2, 1)        # → [B, N, embed_dim]
        x = x + self.pos_embedding    # 加入可学习的位置编码
        return x


class InverseEmbedding(nn.Module):
    def __init__(self, embed_dim=786, output_channels=512, H=7, W=7):
        super(InverseEmbedding, self).__init__()
        self.H = H
        self.W = W
        self.output_channels = output_channels
        self.linear = nn.Linear(embed_dim, output_channels)

    def forward(self, x):
        """
        Args:
            x: [B, N, embed_dim] → 例如 [B, 49, 786]
        Returns:
            [B, C, H, W] → 例如 [B, 512, 7, 7]
        """
        B, N, E = x.shape
        x = self.linear(x)           # [B, 49, 512]
        x = x.permute(0, 2, 1)       # [B, 512, 49]
        x = x.view(B, self.output_channels, self.H, self.W)  # [B, 512, 7, 7]
        return x

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
                ckpt = torch.load(os.path.join('./pretrained/', "MFN_msceleb.pth"))
                print(type(ckpt))
                mixed_feature = torch.load(os.path.join('./pretrained/', "MFN_msceleb.pth"))
            except:
                # Fallback: Disable safe loading entirely
                mixed_feature = torch.load(os.path.join('./pretrained/', "MFN_msceleb.pth"),
                                 weights_only=False, map_location=device)

        self.backbone = nn.Sequential(*list(mixed_feature.children())[:-4])
        self.embedding = PositionalEmbedding(input_channels=512, embed_dim=786, H=7, W=7)
        self.cls_head = ClassifierHead(num_class=num_class, num_head=num_head)
        self.inverse_embedding = InverseEmbedding(embed_dim=786, output_channels=512, H=7, W=7)
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
        # print(f'x_anchor {x_anchor.shape}')
        # Tensor.shape(b, 7, 7, 512)
        cls_base_anchor, x_at_anchor, x_an_head = self.cls_head(x_anchor)
        if x_positive is None:
            print(cls_base_anchor.shape)
            return cls_base_anchor, x_an_head
        '''--------------------------------------------------------------------'''

        '''---------------------------positive-----------------------------------'''
        x_positive = self.backbone(x_positive)
        cls_positive, x_at_positive, x_po_head = self.cls_head(x_positive)
        '''----------------------------------------------------------------------'''

        '''---------------------------negative-----------------------------------'''
        x_negative = self.backbone(x_negative)
        cls_negative, x_at_negative, x_ne_head = self.cls_head(x_negative)
        '''----------------------------------------------------------------------'''

        '''---------------------------negative2----------------------------------'''
        x_negative2 = self.backbone(x_negative2)
        cls_negative2, x_at_negative2, x_ne2_head = self.cls_head(x_negative2)
        '''----------------------------------------------------------------------'''

        #embeded attention map
        x_at_anchor = self.embedding(x_at_anchor)
        x_at_positive = self.embedding(x_at_positive)
        x_at_negative = self.embedding(x_at_negative)
        x_at_negative2 = self.embedding(x_at_negative2)

        x_csa_anchor, x_csa_positive, x_csa_negative, x_csa_negative2, _ = self.cross_similarity_attention(x_at_anchor,
                                                                                    x_at_positive, x_at_negative, x_at_negative2)

        # print(f'x_csa_anchor {x_csa_anchor.shape}')
        # print(f'x_at_positive {x_at_positive.shape}')
        x_at_anchor = x_at_anchor + x_csa_anchor
        x_at_positive = x_at_positive + x_csa_positive
        x_at_negative = x_at_negative + x_csa_negative
        x_at_negative2 = x_at_negative2 + x_csa_negative2

        x_at_anchor = self.inverse_embedding(x_at_anchor)
        x_at_positive = self.inverse_embedding(x_at_positive)
        x_at_negative = self.inverse_embedding(x_at_negative)
        x_at_negative2 = self.inverse_embedding(x_at_negative2)

        # print(f'x_at_anchor {x_at_anchor.shape}')
        x_at_anchor,_, x_at_an_head = self.cls_head(x_at_anchor)
        x_at_positive,_, x_at_po_head = self.cls_head(x_at_positive)
        x_at_negative,_, x_at_ne_head = self.cls_head(x_at_negative)
        x_at_negative2,_,x_at_ne2_head  = self.cls_head(x_at_negative2)

        return (cls_base_anchor, cls_positive, cls_negative, cls_negative2,
                x_at_anchor, x_at_positive, x_at_negative, x_at_negative2, x_an_head, x_po_head, x_ne_head, x_ne2_head,
                x_at_an_head, x_at_po_head, x_at_ne_head, x_at_ne2_head)
