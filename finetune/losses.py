import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def focal_loss(input_values, gamma):
    """Computes the focal loss"""
    p = torch.exp(-input_values)
    loss = (1 - p) ** gamma * input_values
    return loss.mean()

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=0.):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        return focal_loss(F.cross_entropy(input, target, reduction='none', weight=self.weight), self.gamma)

class LDAMLoss(nn.Module):
    
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        self.m_list_li = []
        for cls_nums in cls_num_list:
            li = [int(v) for k,v in cls_nums.items()]
            m_list = 1.0 / np.sqrt(np.sqrt(li))
            print("m_list shape", m_list.shape)  # (2, )
            m_list = m_list * (max_m / np.max(m_list))
            m_list = torch.cuda.FloatTensor(m_list)
            self.m_list_li.append(m_list)
            ## this is list of (weights based on class num list) i.e length 12
        self.task_num = len(self.m_list_li)
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        target = torch.mul(target, 2)
        target.to(torch.int64)
        for i in range(self.task_num): # 12times
            # print(x.shape, target[:,i].shape) #32x12, 32
            tmp = target[:,i].data.view(-1,1).to(torch.int64)
            print( tmp.dtype)
            print(tmp)
            index = torch.zeros_like(x[:,i], dtype=torch.int64)
            
            index = index[np.newaxis]
            index.scatter_(1, tmp, 1) #One hot encoding
            print(index.shape)
            index_float = index.type(torch.cuda.FloatTensor) # 32x2
        
            self.m_list_li[i] = self.m_list_li[i][np.newaxis]
            print(self.m_list_li[i].shape) ## 1 x 2
            batch_m = torch.matmul(self.m_list_li[i], index_float.transpose(0,1)) #1x2, 2,32 -> 1x32
            batch_m = batch_m.view((-1, 1)) # 32x1
            x_m = x - batch_m
    
        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s*output, target, weight=self.weight)


## Original Code
class _LDAMLoss(nn.Module):
    
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1) ##One hot encoding 
        
        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m
    
        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s*output, target, weight=self.weight)