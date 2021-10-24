"""Custom losses."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable


#多分类   sigmoid
class CLASSNetLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.1, class_wgt=[]):
        super(CLASSNetLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        #self.wgt   = [55 , 22]
        self.wgt   = class_wgt

        self.ss = np.sum(self.wgt)

    def forward(self, output, target):
        output = torch.clamp(output, 0.00001, 0.99999)
        

        
        B,C = output.shape
        loss = []
        for b in range(B):
            for c in range(C):
                one_p = output[b,c]
                one_t = target[b,c]

                if one_t>0:
                    loss_one = -1. * (self.ss-self.wgt[c]) * torch.log(one_p) * (1-one_p)**self.gamma
                else:
                    loss_one = -1. * self.wgt[c] * torch.log(1-one_p) * one_p**self.gamma
                loss.append(loss_one/self.ss)

        loss = torch.stack(loss)

        return 10*torch.mean(loss)
        """

        B,C = output.shape
        loss = []
        for b in range(B):
            for c in range(C):
                one_p = output[b,c]
                one_t = target[b,c]

                if one_t>0:
                    loss_one = -1. * (self.ss-self.wgt[c]) * torch.log(one_p) * (1-one_p)**self.gamma
                    loss.append(loss_one/self.ss)

        loss = torch.stack(loss)

        return 10*torch.mean(loss)
        """





"""
#二分类
class CLASSNetLoss(nn.Module):
    def __init__(self, gamma=2,alpha=0.1):
        super(CLASSNetLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.wgt   = [55 , 22]

        self.ss = np.sum(self.wgt)

    def forward(self, output, target):
        output = torch.clamp(output, 0.00001, 0.99999)

        B,C = output.shape
        loss = []
        for b in range(B):
            for c in range(C):
                one_p = output[b,c]
                one_t = target[b,c]

                if one_t>0:
                    loss_one = (self.ss-self.wgt[c]) * (-1. * torch.log(one_p) * (1-one_p)**self.gamma)  
                    loss.append(loss_one)

        loss = torch.stack(loss)

        return torch.mean(loss)

"""





#二分类
class FocalLoss(nn.Module):
    def __init__(self, gamma=2,alpha=0.1):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
    def forward(self, output, target):

        loss  = -(1-self.alpha) * (1-output)**self.gamma * target*torch.log(output) - self.alpha * (1-target) * output**self.gamma * torch.log(1-output)

        return 100*loss.mean()


#二分类
class DiceLoss(nn.Module):
    def __init__(self, smooth=1, p=2):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p

    def forward(self, output, target):
        intersection = torch.sum(torch.mul(output , target)) + self.smooth
        union = torch.sum(torch.pow(output , self.p)) + torch.sum(torch.pow(target , self.p)) + self.smooth

        return 1 - intersection/union

# TODO: optim function
class ICNetLoss(nn.Module):
    """Cross Entropy Loss for ICNet"""
    
    def __init__(self, mode='bce'):
        super(ICNetLoss, self).__init__()
        self.aux_weight = 0.4

        self.mode = mode
        if self.mode == 'bce':
            self.criterion = FocalLoss(gamma=2,alpha=0.25)
            #self.criterion = torch.nn.BCELoss()

        elif self.mode == 'dice':
            self.criterion = DiceLoss(smooth=1, p=2)

    def forward(self, out_16, out_8, out_4, out_1, target):
        ############## 1/16    1/8    1/4     1       1


        target = target.float()
        target4  = F.interpolate(target, out_4.size()[2:],  mode='bilinear', align_corners=True).long()
        target8  = F.interpolate(target, out_8.size()[2:],  mode='bilinear', align_corners=True).long()
        target16 = F.interpolate(target, out_16.size()[2:], mode='bilinear', align_corners=True).long()
  
        loss1 = self.criterion(out_4.view(-1),  target4.view(-1).float())
        loss2 = self.criterion(out_8.view(-1),  target8.view(-1).float())
        loss3 = self.criterion(out_16.view(-1), target16.view(-1).float())
        
        return loss1 + loss2 * self.aux_weight + loss3 * self.aux_weight

