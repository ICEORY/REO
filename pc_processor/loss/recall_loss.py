import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class RecallLoss(nn.Module):
    def __init__(self):
        super(RecallLoss, self).__init__()

    def forward(self, x, target, mask=None):
        """compute focal loss
        x: N C or NCHW
        target: N, or NHW

        Args:
            x ([type]): [description]
            target ([type]): [description]
        """

        if mask is not None:
            target = target * mask
            # if mask.dim() == x.dim() -1:
            #     mask = mask.unsqueeze(1)
            # x = x * mask

        if x.dim() > 2:
            pred = x.view(x.size(0), x.size(1), -1)
            pred = pred.transpose(1, 2)
            pred = pred.contiguous().view(-1, x.size(1))
        else:
            pred = x

        n_classes = x.size(1)
        n_batch = x.size(0)
        target = F.one_hot(target, num_classes=n_classes) # shape: NC or NHWC
        if target.dim() > 2:
            target = target.view(-1, n_classes) # NC

        # reshape to NC
        if x.dim() > 2:
            pred = x.view(x.size(0), x.size(1), -1)
            pred = pred.transpose(1, 2)
            pred = pred.contiguous().view(-1, x.size(1))
        else:
            pred = x
        
        pred_sigmoid = F.sigmoid(pred)
        if mask is not None:
            mask = mask.view(-1, 1) # N, 1
            pred_sigmoid = pred_sigmoid * mask

        recall_loss = (target - pred_sigmoid * target).mean() 
        return recall_loss