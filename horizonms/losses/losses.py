import torch
from .losses_func import smooth_l1_loss, weak_cross_entropy
from .. import LOSSES


__all__ = ("WeakCrossEntropyLoss", "CDRSmoothL1Loss", "PseudoCrossEntropyLoss", "PseudoPositiveCrossEntropyLoss")


@LOSSES.register_module()
class WeakCrossEntropyLoss():
    def __init__(self, mode='all', w_alpha=4.0, toppk=-1,
    focal_params={'alpha':0.25, 'gamma':2.0}, epsilon=1e-6):
        super(WeakCrossEntropyLoss, self).__init__()
        self.mode = mode
        self.w_alpha = w_alpha
        self.toppk = toppk
        self.focal_params = focal_params
        self.epsilon = epsilon
        
    def __call__(self, gt_mask, gt_boxes, ypred):
        loss = weak_cross_entropy(ypred, gt_mask, gt_boxes, mode=self.mode, 
                                  w_alpha=self.w_alpha, toppk=self.toppk, 
                                  focal_params=self.focal_params, epsilon=self.epsilon)
        return loss


@LOSSES.register_module()
class CDRSmoothL1Loss():
    def __init__(self, calculate_ratio=True, sigma=3.0, size_average=True, smooth=1e-10):
        super(CDRSmoothL1Loss, self).__init__()
        self.calculate_ratio = calculate_ratio
        self.sigma = sigma
        self.size_average = size_average
        self.smooth = smooth
        
    def __call__(self, cup_gt, disc_gt, cup_pd, disc_pd):
        cdr_gt = cup_gt / (disc_gt + self.smooth)
        if self.calculate_ratio:
            cdr_pd = cup_pd / (disc_pd + self.smooth)
            loss = smooth_l1_loss(cdr_gt, cdr_pd, sigma=self.sigma, size_average=self.size_average)
        else:
            pd = cup_pd / (disc_gt + self.smooth)
            gt = disc_pd * cdr_gt / (disc_gt + self.smooth)
            loss = smooth_l1_loss(pd, gt, sigma=self.sigma, size_average=self.size_average)
        return loss


@LOSSES.register_module()
class PseudoCrossEntropyLoss():
    def __init__(self, mode='all', soft=True, threshold=0.5, epsilon=1e-6):
        self.mode = mode
        self.soft = soft
        self.threshold = threshold
        self.epsilon = epsilon

    def __call__(self, pseudo_true, mask, ypred):
        if self.soft:
            pseudo_pos = pseudo_true*mask*(pseudo_true>self.threshold)
        else:
            pseudo_pos = mask*(pseudo_true>self.threshold)
        ypred = torch.clamp(ypred, self.epsilon, 1-self.epsilon)
        loss_pos = -pseudo_pos*torch.log(ypred)
        loss_neg = -(1-mask)*torch.log(1-ypred)

        loss_pos = torch.sum(loss_pos, dim=(0,2,3))
        loss_neg = torch.sum(loss_neg, dim=(0,2,3))
        nb_pos = torch.sum(pseudo_pos, dim=(0,2,3))
        nb_neg = torch.sum(1-mask, dim=(0,2,3))

        if self.mode=='all':
            loss = (loss_pos+loss_neg)/(nb_pos+nb_neg)
        elif self.mode=='balance':
            loss = (loss_pos/(nb_pos+self.epsilon)+loss_neg/nb_neg)/2
        return loss


@LOSSES.register_module()
class PseudoPositiveCrossEntropyLoss():
    def __init__(self, soft=True, threshold=0.5, epsilon=1e-6):
        self.soft = soft
        self.threshold = threshold
        self.epsilon = epsilon

    def __call__(self, pseudo_true, mask, ypred):
        if self.soft:
            pseudo_pos = pseudo_true*mask*(pseudo_true>self.threshold)
        else:
            pseudo_pos = mask*(pseudo_true>self.threshold)
        ypred = torch.clamp(ypred, self.epsilon, 1-self.epsilon)
        loss = -pseudo_pos*torch.log(ypred)
        loss = torch.mean(loss, dim=(0,2,3))
        return loss
