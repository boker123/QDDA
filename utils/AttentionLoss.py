import sys
from torch import nn
import torch.nn.functional as F
eps = sys.float_info.epsilon


class AttentionLoss(nn.Module):
    def __init__(self, ):
        super(AttentionLoss, self).__init__()

    def forward(self, x):
        num_head = len(x)
        loss = 0
        cnt = 0
        if num_head > 1:
            for i in range(num_head-1):
                for j in range(i+1, num_head):
                    mse = F.mse_loss(x[i], x[j])
                    cnt = cnt+1
                    loss = loss+mse
            loss = cnt/(loss + eps)
        else:
            loss = 0
        return loss