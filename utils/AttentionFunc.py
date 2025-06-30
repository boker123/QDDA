import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import shutil
import math


import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt



'''-----------------------  QCS  SD  ------------------------'''
def Attn_QCS_SD(QK1, QK2, QK3, QK4, k):
    B, N, C = QK1.shape
    k = k + 1

    D_2x1 = torch.cdist(QK2, QK1, p=2)  # B x N2 x C , B x N1 x C --> B x N2 x N1
    D_2x1 = D_2x1 - torch.min(D_2x1)
    S_2x1 = torch.max(D_2x1) - D_2x1

    D_4x3 = torch.cdist(QK4, QK3, p=2)  # B x N4 x C , B x N3 x C --> B x N4 x N3
    D_4x3 = D_4x3 - torch.min(D_4x3)
    S_4x3 = torch.max(D_4x3) - D_4x3

    D_3x1 = torch.cdist(QK3, QK1, p=2)  # B x N3 x N1
    D_4x2 = torch.cdist(QK4, QK2, p=2)  # B x N4 x N2
    D_3x1 = D_3x1 - torch.min(D_3x1)
    D_4x2 = D_4x2 - torch.min(D_4x2)

    ######################## K1 ########################
    S_2x1_n1 = F.normalize(S_2x1, p=2, dim=2)
    S1 = torch.sum(S_2x1_n1, dim=1)  # B x N1
    ''''''''''''''''''''''''''
    D_3x1_n1 = F.normalize(D_3x1, p=2, dim=2)
    D1 = torch.sum(D_3x1_n1, dim=1)  # B x N1

    ''''''''''''''''''''''''''
    SD1 = S1 + k * D1
    cross_map1 = F.softmax(SD1, dim=-1)

    ######################## Q4 ########################
    S_4x3_n4 = F.normalize(S_4x3, p=2, dim=1)
    S4 = torch.sum(S_4x3_n4, dim=2)  # B x N4
    ''''''''''''''''''''''''''
    D_4x2_n4 = F.normalize(D_4x2, p=2, dim=1)
    D4 = torch.sum(D_4x2_n4, dim=2)  # B x N4

    ''''''''''''''''''''''''''
    SD4 = S4 + k * D4
    cross_map4 = F.softmax(SD4, dim=-1)

    ######################## K2 ########################
    S_1x2 = S_2x1.transpose(1, 2)  # B x N1 x N2
    S_1x2_n2 = F.normalize(S_1x2, p=2, dim=2)
    S2 = torch.sum(S_1x2_n2, dim=1)  # B x N2
    ''''''''''''''''''''''''''
    D_4x2_n2 = F.normalize(D_4x2, p=2, dim=2)
    D2 = torch.sum(D_4x2_n2, dim=1)  # B x N2
    ''''''''''''''''''''''''''
    SD2 = S2 + k * D2
    cross_map2 = F.softmax(SD2, dim=-1)

    ######################## Q3 ########################
    S_3x4 = S_4x3.transpose(1, 2)  # B x N3 x N4
    S_3x4_n3 = F.normalize(S_3x4, p=2, dim=1)
    S3 = torch.sum(S_3x4_n3, dim=2)  # B x N3
    ''''''''''''''''''''''''''
    D_3x1_n3 = F.normalize(D_3x1, p=2, dim=1)
    D3 = torch.sum(D_3x1_n3, dim=2)  # B x N3
    ''''''''''''''''''''''''''
    SD3 = S3 + k * D3
    cross_map3 = F.softmax(SD3, dim=-1)

    return cross_map1, cross_map2, cross_map3, cross_map4