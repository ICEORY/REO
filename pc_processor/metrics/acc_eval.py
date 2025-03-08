#!/usr/bin/env python3

# This file is covered by the LICENSE file in the root of this project.

import torch 
from ..utils import AverageMeter

class AccEval(object):
    def __init__(self, topk=(1, ), is_distributed=False):
        self.topk = topk 
        self.is_distributed = is_distributed
        self.topk_meter = []
        self.reset()
    
    def reset(self):
        self.topk_meter = []
        for _ in range(len(self.topk)):
            self.topk_meter.append(AverageMeter())

    def addBatch(self, output, target):
        maxk = max(self.topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        for i, k in enumerate(self.topk):
            correct_k = correct[:k].float().sum()

            if self.is_distributed:
                correct_k = correct_k.cuda()
                total_batch_size = torch.Tensor([batch_size]).cuda()
                torch.distributed.barrier()
                torch.distributed.all_reduce(correct_k)
                torch.distributed.all_reduce(total_batch_size)
                
                correct_k = correct_k.item()
                total_batch_size = total_batch_size.item()
            else:
                total_batch_size = batch_size    
            
            self.topk_meter[i].update(correct_k, total_batch_size)

            # acc = correct_k * 100.0 / total_batch_size
            # res.append(acc)

    def getAcc(self):
        acc = []
        for meter in self.topk_meter:
            acc.append(meter.avg)
        return acc
