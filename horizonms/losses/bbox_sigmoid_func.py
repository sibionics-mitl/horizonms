import copy
import math
import torch
import torch.nn.functional as F
from .parallel_transform import parallel_transform
from .polar_transform import polar_transform


def mil_unary_baseline_sigmoid_loss(ypred, mask, gt_boxes_xxyy,
        loss_mode='ce_all', focal_params={'alpha':0.25, 'gamma':2.0}, epsilon=1e-6):
    """ Compute the mil baseline unary loss. 

    Args:
        ypred: Tensor of predicted data from the network with shape (B, C, W, H).
        mask:  Tensor of mask with shape (B, C, W, H), bounding box regions with value 1 and 0 otherwise.
        gt_boxes_xxyy: Tensor of boxes with (N, 6), where N is the number of bouding boxes in the batch,
                    the 6 elements of each row are [nb_img, class, x1, y1, x2, y2]

    Returns:
        losses with shape (C,) for each category

    Reference:
        Wang, J. and Xia, B., 2021, September. Bounding box tightness prior
        for weakly supervised image segmentation. In International Conference on 
        Medical Image Computing and Computer-Assisted Intervention (pp. 526-536). 
        Springer, Cham.
    """
    assert loss_mode in ['ce_all', 'ce_balance', 'focal']
    ypred = torch.clamp(ypred, epsilon, 1-epsilon)
    num_classes = ypred.shape[1]
    ypred_pos = {c:[] for c in range(num_classes)}
    for nb_ob in range(gt_boxes_xxyy.shape[0]):
        nb_img = gt_boxes_xxyy[nb_ob,0]
        c = gt_boxes_xxyy[nb_ob,1].item()
        box = gt_boxes_xxyy[nb_ob,2:]
        pred = ypred[nb_img,c,box[1]:box[3]+1,box[0]:box[2]+1]
        if pred.numel() == 0:
            continue
        ypred_pos[c].append(torch.max(pred, dim=0)[0])
        ypred_pos[c].append(torch.max(pred, dim=1)[0])
    
    v1 = torch.max(ypred*(1-mask), dim=2)[0]
    v2 = torch.max(ypred*(1-mask), dim=3)[0]
    ypred_neg = torch.cat([v1,v2], dim=-1).permute(1,0,2)
    ypred_neg = torch.reshape(ypred_neg, (ypred_neg.shape[0],-1))

    if loss_mode == 'focal':
        alpha = focal_params['alpha']
        gamma = focal_params['gamma']
        losses = torch.zeros((num_classes,), dtype=ypred.dtype, device=ypred.device)
        for c in range(num_classes):
            bce_neg = -(1-alpha)*(ypred_neg**gamma)*torch.log(1-ypred_neg)
            if len(ypred_pos[c])>0:
                y_pos = torch.cat(ypred_pos[c], dim=0)
                bce_pos = -alpha*((1-y_pos)**gamma)*torch.log(y_pos)
                loss = (bce_neg.sum()+bce_pos.sum())/len(bce_pos)
            else:
                loss = bce_neg.sum()
            losses[c] = loss
    else:
        losses = torch.zeros((num_classes,), dtype=ypred.dtype, device=ypred.device)
        for c in range(num_classes):
            bce_neg = -torch.log(1-ypred_neg[c])
            if len(ypred_pos[c])>0:
                bce_pos = -torch.log(torch.cat(ypred_pos[c], dim=0))
                if loss_mode == 'ce_all':
                    loss = (bce_pos.sum()+bce_neg.sum())/(len(bce_pos)+len(bce_neg))
                elif loss_mode == 'ce_balance':
                    loss = (bce_pos.mean()+bce_neg.mean())/2
            else:
                loss = bce_neg.mean()
            losses[c] = loss
    return losses


def mil_approx_unary_baseline_sigmoid_loss(ypred, mask, gt_boxes_xxyy, 
        loss_mode='ce_all', approx_method='softmax', approx_alpha=4, 
        focal_params={'alpha':0.25, 'gamma':2.0}, epsilon=1e-6):
    """ Compute the mil baseline unary loss while applying smooth maximum function. 

    Args:
        ypred: Tensor of predicted data from the network with shape (B, C, W, H).
        mask:  Tensor of mask with shape (B, C, W, H), bounding box regions with value 1 and 0 otherwise.
        gt_boxes_xxyy: Tensor of boxes with (N, 6), where N is the number of bouding boxes in the batch,
                    the 6 elements of each row are [nb_img, class, x1, y1, x2, y2]

    Returns:
        losses with shape (C,) for each category

    Reference:
        Wang, J. and Xia, B., 2021, September. Bounding box tightness prior
        for weakly supervised image segmentation. In International Conference on 
        Medical Image Computing and Computer-Assisted Intervention (pp. 526-536). 
        Springer, Cham.
    """
    assert loss_mode in ['ce_all', 'ce_balance', 'focal']
    assert approx_method in ['softmax', 'quasimax']
    ypred = torch.clamp(ypred, epsilon, 1-epsilon)
    ypred_g = torch.exp(approx_alpha*ypred) # smooth maximum approximation
    num_classes = ypred.shape[1]
    ypred_pos = {c:[] for c in range(num_classes)}
    for nb_ob in range(gt_boxes_xxyy.shape[0]):
        nb_img = gt_boxes_xxyy[nb_ob,0]
        c = gt_boxes_xxyy[nb_ob,1].item()
        box = gt_boxes_xxyy[nb_ob,2:]
        pred = ypred_g[nb_img,c,box[1]:box[3]+1,box[0]:box[2]+1]
        if approx_method == 'softmax':
            pd_org = ypred[nb_img,c,box[1]:box[3]+1,box[0]:box[2]+1]
            prob0 = torch.sum(pd_org*pred,dim=0)/torch.sum(pred,dim=0)
            prob1 = torch.sum(pd_org*pred,dim=1)/torch.sum(pred,dim=1)
        elif approx_method == 'quasimax':
            msk = mask[nb_img,c,box[1]:box[3]+1,box[0]:box[2]+1]
            prob0 = torch.log(torch.sum(pred,dim=0))/approx_alpha - torch.log(torch.sum(msk,dim=0))/approx_alpha
            prob1 = torch.log(torch.sum(pred,dim=1))/approx_alpha - torch.log(torch.sum(msk,dim=1))/approx_alpha
        ypred_pos[c].append(prob0)
        ypred_pos[c].append(prob1)
    
    ## for negative class
    if approx_method == 'softmax':
        v1 = torch.sum(ypred*ypred_g*(1-mask), dim=2)/torch.sum(ypred_g*(1-mask), dim=2)
        v2 = torch.sum(ypred*ypred_g*(1-mask), dim=3)/torch.sum(ypred_g*(1-mask), dim=3)
    elif approx_method == 'quasimax':
        v1 = torch.log(torch.sum(ypred_g*(1-mask), dim=2))/approx_alpha - torch.log(torch.sum(1-mask, dim=2))/approx_alpha
        v2 = torch.log(torch.sum(ypred_g*(1-mask), dim=3))/approx_alpha - torch.log(torch.sum(1-mask, dim=3))/approx_alpha
    ypred_neg = torch.cat([v1,v2], dim=-1).permute(1,0,2)
    ypred_neg = torch.reshape(ypred_neg, (ypred_neg.shape[0],-1))

    if loss_mode == 'focal':
        alpha = focal_params['alpha']
        gamma = focal_params['gamma']
        losses = torch.zeros((num_classes,), dtype=ypred.dtype, device=ypred.device)
        for c in range(num_classes):
            bce_neg = -(1-alpha)*(ypred_neg**gamma)*torch.log(1-ypred_neg)
            if len(ypred_pos[c])>0:
                y_pos = torch.cat(ypred_pos[c], dim=0)
                bce_pos = -alpha*((1-y_pos)**gamma)*torch.log(y_pos)
                loss = (bce_neg.sum()+bce_pos.sum())/len(bce_pos)
            else:
                loss = bce_neg.sum()
            losses[c] = loss
    else:
        losses = torch.zeros((num_classes,), dtype=ypred.dtype, device=ypred.device)
        for c in range(num_classes):
            neg = torch.clamp(1-ypred_neg[c], epsilon, 1-epsilon)
            bce_neg = -torch.log(neg)
            if len(ypred_pos[c])>0:
                pos = torch.clamp(torch.cat(ypred_pos[c], dim=0), epsilon, 1-epsilon)
                bce_pos = -torch.log(pos)
                if loss_mode == 'ce_all':
                    loss = (bce_pos.sum()+bce_neg.sum())/(len(bce_pos)+len(bce_neg))
                elif loss_mode == 'ce_balance':
                    loss = (bce_pos.mean()+bce_neg.mean())/2
            else:
                loss = bce_neg.mean()
            losses[c] = loss
    return losses


def mil_unary_baseline_pos_generalized_neg_sigmoid_loss(ypred, mask, gt_boxes_xxyy, 
        loss_mode='ce_all', focal_params={'alpha':0.25, 'gamma':2.0}, epsilon=1e-6):
    """ Compute the mil baseline unary loss, in which the generalized negative bag is used.

    Args:
        ypred: Tensor of predicted data from the network with shape (B, C, W, H).
        mask:  Tensor of mask with shape (B, C, W, H), bounding box regions with value 1 and 0 otherwise.
        gt_boxes_xxyy: Tensor of boxes with (N, 6), where N is the number of bouding boxes in the batch,
                    the 6 elements of each row are [nb_img, class, x1, y1, x2, y2]

    Returns:
        losses with shape (C,) for each category

    Reference:
        Wang, J. and Xia, B., 2021, September. Bounding box tightness prior
        for weakly supervised image segmentation. In International Conference on 
        Medical Image Computing and Computer-Assisted Intervention (pp. 526-536). 
        Springer, Cham.
    """
    assert loss_mode in ['ce_all', 'ce_balance', 'focal']
    ypred = torch.clamp(ypred, epsilon, 1-epsilon)
    num_classes = ypred.shape[1]
    ypred_pos = {c:[] for c in range(num_classes)}
    for nb_ob in range(gt_boxes_xxyy.shape[0]):
        nb_img = gt_boxes_xxyy[nb_ob,0]
        c = gt_boxes_xxyy[nb_ob,1].item()
        box = gt_boxes_xxyy[nb_ob,2:]
        pred = ypred[nb_img,c,box[1]:box[3]+1,box[0]:box[2]+1]
        if pred.numel() == 0:
            continue
        ypred_pos[c].append(torch.max(pred, dim=0)[0])
        ypred_pos[c].append(torch.max(pred, dim=1)[0])
        
    if loss_mode == 'focal':
        alpha = focal_params['alpha']
        gamma = focal_params['gamma']
        losses = torch.zeros((num_classes,), dtype=ypred.dtype, device=ypred.device)
        for c in range(num_classes):
            y_neg = ypred[:,c,:,:]
            y_neg = y_neg[mask[:,c,:,:]<0.5]
            bce_neg = -(1-alpha)*(y_neg**gamma)*torch.log(1-y_neg)
            if len(ypred_pos[c])>0:
                y_pos = torch.cat(ypred_pos[c], dim=0)
                y_pos = torch.clamp(y_pos, epsilon, 1-epsilon)
                bce_pos = -alpha*((1-y_pos)**gamma)*torch.log(y_pos)
                loss = (bce_neg.sum()+bce_pos.sum())/len(bce_pos)
            else:
                loss = bce_neg.sum()
            losses[c] = loss
    else:
        losses = torch.zeros((num_classes,), dtype=ypred.dtype, device=ypred.device)
        for c in range(num_classes):
            y_neg = ypred[:,c,:,:]
            y_neg = y_neg[mask[:,c,:,:]<0.5]
            bce_neg = -torch.log(1-y_neg[c])
            if len(ypred_pos[c])>0:
                bce_pos = -torch.log(torch.cat(ypred_pos[c], dim=0))
                if loss_mode == 'ce_all':
                    loss = (bce_pos.sum()+bce_neg.sum())/(len(bce_pos)+len(bce_neg))
                elif loss_mode == 'ce_balance':
                    loss = (bce_pos.mean()+bce_neg.mean())/2
            else:
                loss = bce_neg.mean()
            losses[c] = loss            
    return losses


def mil_approx_unary_baseline_pos_generalized_neg_sigmoid_loss(ypred, mask,
        gt_boxes_xxyy, loss_mode='ce_all', approx_method='softmax', approx_alpha=4, 
        focal_params={'alpha':0.25, 'gamma':2.0}, epsilon=1e-6):
    """ Compute the mil baseline unary loss while applying smooth maximum function,
        in which the generalized negative bag is used.

    Args:
        ypred: Tensor of predicted data from the network with shape (B, C, W, H).
        mask:  Tensor of mask with shape (B, C, W, H), bounding box regions with value 1 and 0 otherwise.
        gt_boxes_xxyy: Tensor of boxes with (N, 6), where N is the number of bouding boxes in the batch,
                    the 6 elements of each row are [nb_img, class, x1, y1, x2, y2]

    Returns:
        losses with shape (C,) for each category

    Reference:
        Wang, J. and Xia, B., 2021, September. Bounding box tightness prior
        for weakly supervised image segmentation. In International Conference on 
        Medical Image Computing and Computer-Assisted Intervention (pp. 526-536). 
        Springer, Cham.
    """
    assert loss_mode in ['ce_all', 'ce_balance', 'focal']
    assert approx_method in ['softmax', 'quasimax']
    ypred = torch.clamp(ypred, epsilon, 1-epsilon)
    ypred_g = torch.exp(approx_alpha*ypred) # smooth maximum approximation
    num_classes = ypred.shape[1]
    ypred_pos = {c:[] for c in range(num_classes)}
    for nb_ob in range(gt_boxes_xxyy.shape[0]):
        nb_img = gt_boxes_xxyy[nb_ob,0]
        c = gt_boxes_xxyy[nb_ob,1].item()
        box = gt_boxes_xxyy[nb_ob,2:]
        pred = ypred_g[nb_img,c,box[1]:box[3]+1,box[0]:box[2]+1]
        if approx_method == 'softmax':
            pd_org = ypred[nb_img,c,box[1]:box[3]+1,box[0]:box[2]+1]
            prob0 = torch.sum(pd_org*pred,dim=0)/torch.sum(pred,dim=0)
            prob1 = torch.sum(pd_org*pred,dim=1)/torch.sum(pred,dim=1)
        elif approx_method == 'quasimax':
            msk = mask[nb_img,c,box[1]:box[3]+1,box[0]:box[2]+1]
            prob0 = torch.log(torch.sum(pred,dim=0))/approx_alpha - torch.log(torch.sum(msk,dim=0))/approx_alpha
            prob1 = torch.log(torch.sum(pred,dim=1))/approx_alpha - torch.log(torch.sum(msk,dim=1))/approx_alpha
        ypred_pos[c].append(prob0)
        ypred_pos[c].append(prob1)
    
    if loss_mode == 'focal':
        alpha = focal_params['alpha']
        gamma = focal_params['gamma']
        losses = torch.zeros((num_classes,), dtype=ypred.dtype, device=ypred.device)
        for c in range(num_classes):
            y_neg = ypred[:,c,:,:]
            y_neg = y_neg[mask[:,c,:,:]<0.5]
            bce_neg = -(1-alpha)*(y_neg**gamma)*torch.log(1-y_neg)
            if len(ypred_pos[c])>0:
                y_pos = torch.cat(ypred_pos[c], dim=0)
                y_pos = torch.clamp(y_pos, epsilon, 1-epsilon)
                bce_pos = -alpha*((1-y_pos)**gamma)*torch.log(y_pos)
                loss = (bce_neg.sum()+bce_pos.sum())/len(bce_pos)
            else:
                loss = bce_neg.sum()
            losses[c] = loss
    else:
        losses = torch.zeros((num_classes,), dtype=ypred.dtype, device=ypred.device)
        for c in range(num_classes):
            y_neg = ypred[:,c,:,:]
            y_neg = y_neg[mask[:,c,:,:]<0.5]
            bce_neg = -torch.log(1-y_neg[c])
            if len(ypred_pos[c])>0:
                bce_pos = -torch.log(torch.cat(ypred_pos[c], dim=0))
                if loss_mode == 'ce_all':
                    loss = (bce_pos.sum()+bce_neg.sum())/(len(bce_pos)+len(bce_neg))
                elif loss_mode == 'ce_balance':
                    loss = (bce_pos.mean()+bce_neg.mean())/2
            else:
                loss = bce_neg.mean()
            losses[c] = loss      
    return losses


def mil_unary_parallel_sigmoid_loss(ypred, mask, gt_boxes_cr, 
        loss_mode='ce_all', angle_params=(60,-61,30), 
        focal_params={'alpha':0.25, 'gamma':2.0, 'sampling_prob':1.0}, 
        epsilon=1e-6):
    """ Compute the mil generalized unary loss while no smooth maximum function is applied.

    Args:
        ypred: Tensor of predicted data from the network with shape (B, C, W, H).
        mask:  Tensor of mask with shape (B, C, W, H), bounding box regions with value 1 and 0 otherwise.
        gt_boxes_cr: Tensor of boxes with (N, 5), where N is the number of bouding boxes in the batch,
                    the 5 elements of each row are [nb_img, class, center_x, center_r, radius]

    Returns:
        losses with shape (C,) for each category

    Reference:
        Wang, J. and Xia, B., 2021, September. Bounding box tightness prior
        for weakly supervised image segmentation. In International Conference on 
        Medical Image Computing and Computer-Assisted Intervention (pp. 526-536). 
        Springer, Cham.
    """
    assert loss_mode in ['ce_all', 'ce_balance', 'focal']
    ypred = torch.clamp(ypred, epsilon, 1-epsilon)
    num_classes = ypred.shape[1]
    ob_img_index = gt_boxes_cr[:,0].type(torch.int32)
    ob_class_index = gt_boxes_cr[:,1].type(torch.int32)
    ob_gt_boxes_cr = gt_boxes_cr[:,2:]
    ypred_pos = {c:[] for c in range(num_classes)}
    for nb_ob in range(gt_boxes_cr.shape[0]):
        nb_img = ob_img_index[nb_ob]
        c = ob_class_index[nb_ob]
        extra = 5
        cx,cy,r = ob_gt_boxes_cr[nb_ob,:].type(torch.int32)
        r = r + extra
        xmin = torch.clamp(cx-r,0)
        ymin = torch.clamp(cy-r,0)
        pred = ypred[nb_img,c,ymin:cy+r+1,xmin:cx+r+1][None,:,:]
        msk  = mask[nb_img,c,ymin:cy+r+1,xmin:cx+r+1][None,:,:]

        index = torch.nonzero(msk[0]>0.5, as_tuple=True)
        y0,y1 = index[0].min(), index[0].max()
        x0,x1 = index[1].min(), index[1].max()
        box_h = y1-y0+1
        box_w = x1-x0+1

        parallel_angle_params = list(range(angle_params[0],angle_params[1],angle_params[2]))
        
        for angle in parallel_angle_params:
            pred_parallel = parallel_transform(pred, box_h, box_w, angle, is_mask=False)
            msk0, msk1  = parallel_transform(msk, box_h, box_w, angle, is_mask=True)
            pred_parallel0 = pred_parallel*msk0
            pred_parallel1 = pred_parallel*msk1
            flag0 = torch.sum(msk0[0], dim=0)>0.5
            prob0 = torch.max(pred_parallel0[0], dim=0)[0]
            prob0 = prob0[flag0]
            flag1 = torch.sum(msk1[0], dim=1)>0.5
            prob1 = torch.max(pred_parallel1[0], dim=1)[0]
            prob1 = prob1[flag1]
            if len(prob0)>0:
                ypred_pos[c.item()].append(prob0)
            if len(prob1)>0:
                ypred_pos[c.item()].append(prob1)

    if loss_mode == 'focal':
        alpha = focal_params['alpha']
        gamma = focal_params['gamma']
        losses = torch.zeros((num_classes,), dtype=ypred.dtype, device=ypred.device)
        for c in range(num_classes):
            y_neg = ypred[:,c,:,:]
            y_neg = y_neg[mask[:,c,:,:]<0.5]
            bce_neg = -(1-alpha)*(y_neg**gamma)*torch.log(1-y_neg)
            if len(ypred_pos[c])>0:
                y_pos = torch.cat(ypred_pos[c], dim=0)
                y_pos = torch.clamp(y_pos, epsilon, 1-epsilon)
                bce_pos = -alpha*((1-y_pos)**gamma)*torch.log(y_pos)
                loss = (bce_neg.sum()+bce_pos.sum())/len(bce_pos)
            else:
                loss = bce_neg.sum()
            losses[c] = loss
    else:
        losses = torch.zeros((num_classes,), dtype=ypred.dtype, device=ypred.device)
        for c in range(num_classes):
            y_neg = ypred[:,c,:,:]
            y_neg = y_neg[mask[:,c,:,:]<0.5]
            bce_neg = -torch.log(1-y_neg[c])
            if len(ypred_pos[c])>0:
                pred = torch.clamp(torch.cat(ypred_pos[c], dim=0), epsilon, 1-epsilon)
                bce_pos = -torch.log(pred)
                if loss_mode == 'ce_all':
                    loss = (bce_pos.sum()+bce_neg.sum())/(len(bce_pos)+len(bce_neg))
                elif loss_mode == 'ce_balance':
                    loss = (bce_pos.mean()+bce_neg.mean())/2
            else:
                loss = bce_neg.mean()
            losses[c] = loss

    return losses


def mil_approx_unary_parallel_sigmoid_loss(ypred, mask, gt_boxes_cr, 
        loss_mode='focal', angle_params=(60,-61,30),
        approx_method='softmax', approx_alpha=4, 
        focal_params={'alpha':0.25, 'gamma':2.0}, epsilon=1e-6):
    """ Compute the mil generalized unary loss.

    Args:
        ypred: Tensor of predicted data from the network with shape (B, C, W, H).
        mask:  Tensor of mask with shape (B, C, W, H), bounding box regions with value 1 and 0 otherwise.
        gt_boxes_cr: Tensor of boxes with (N, 5), where N is the number of bouding boxes in the batch,
                    the 5 elements of each row are [nb_img, class, center_x, center_r, radius]

    Returns:
        losses with shape (C,) for each category

    Reference:
        Wang, J. and Xia, B., 2021, September. Bounding box tightness prior
        for weakly supervised image segmentation. In International Conference on 
        Medical Image Computing and Computer-Assisted Intervention (pp. 526-536). 
        Springer, Cham.
    """
    assert loss_mode in ['ce_all', 'ce_balance', 'focal']
    assert approx_method in ['softmax', 'quasimax']
    parallel_angle_params = list(range(angle_params[0],angle_params[1],angle_params[2]))
    ypred = torch.clamp(ypred, epsilon, 1-epsilon)
    num_classes = ypred.shape[1]
    ob_img_index = gt_boxes_cr[:,0].type(torch.int32)
    ob_class_index = gt_boxes_cr[:,1].type(torch.int32)
    ob_gt_boxes_cr = gt_boxes_cr[:,2:]
    ypred_pos = {c:[] for c in range(num_classes)}
    for nb_ob in range(gt_boxes_cr.shape[0]):
        nb_img = ob_img_index[nb_ob]
        c = ob_class_index[nb_ob]
        extra = 5
        cx,cy,r = ob_gt_boxes_cr[nb_ob,:].type(torch.int32)
        r = r + extra
        xmin = torch.clamp(cx-r,0)
        ymin = torch.clamp(cy-r,0)
        pred = ypred[nb_img,c,ymin:cy+r+1,xmin:cx+r+1][None,:,:]
        msk  = mask[nb_img,c,ymin:cy+r+1,xmin:cx+r+1][None,:,:]

        index = torch.nonzero(msk[0]>0.5, as_tuple=True)
        y0,y1 = index[0].min(), index[0].max()
        x0,x1 = index[1].min(), index[1].max()
        box_h = y1-y0+1
        box_w = x1-x0+1
        
        for angle in parallel_angle_params:
            pred_parallel = parallel_transform(pred, box_h, box_w, angle, is_mask=False)
            msk0, msk1  = parallel_transform(msk, box_h, box_w, angle, is_mask=True)
            pred_parallel = pred_parallel[0]
            msk0 = msk0[0]>0.5
            msk1 = msk1[0]>0.5
            flag0 = torch.sum(msk0, dim=0)>0.5
            flag1 = torch.sum(msk1, dim=1)>0.5
            pred_parallel0 = pred_parallel[:,flag0]
            pred_parallel1 = pred_parallel[flag1,:]
            msk0 = msk0[:,flag0]
            msk1 = msk1[flag1,:]
            
            if torch.sum(flag0)>0.5:
                if approx_method == 'softmax':
                    w = torch.exp(approx_alpha*pred_parallel0)
                    prob0 = torch.sum(pred_parallel0*w*msk0,dim=0)/torch.sum(w*msk0,dim=0)
                elif approx_method == 'quasimax':
                    w = torch.exp(approx_alpha*pred_parallel0)
                    prob0 = torch.log(torch.sum(w*msk0,dim=0))/approx_alpha - torch.log(torch.sum(msk0, dim=0))/approx_alpha
                ypred_pos[c.item()].append(prob0)
            if torch.sum(flag1)>0.5:
                if approx_method == 'softmax':
                    w = torch.exp(approx_alpha*pred_parallel1)
                    prob1 = torch.sum(pred_parallel1*w*msk1,dim=1)/torch.sum(w*msk1,dim=1)
                elif approx_method == 'quasimax':
                    w = torch.exp(approx_alpha*pred_parallel1)
                    prob1 = torch.log(torch.sum(w*msk1,dim=1))/approx_alpha - torch.log(torch.sum(msk1,dim=1))/approx_alpha
                ypred_pos[c.item()].append(prob1)

    if loss_mode == 'focal':
        alpha = focal_params['alpha']
        gamma = focal_params['gamma']
        losses = torch.zeros((num_classes,), dtype=ypred.dtype, device=ypred.device)
        for c in range(num_classes):
            y_neg = ypred[:,c,:,:]
            y_neg = y_neg[mask[:,c,:,:]<0.5]
            bce_neg = -(1-alpha)*(y_neg**gamma)*torch.log(1-y_neg)
            if len(ypred_pos[c])>0:
                y_pos = torch.cat(ypred_pos[c], dim=0)
                y_pos = torch.clamp(y_pos, epsilon, 1-epsilon)
                bce_pos = -alpha*((1-y_pos)**gamma)*torch.log(y_pos)
                loss = (bce_neg.sum()+bce_pos.sum())/len(bce_pos)
            else:
                loss = bce_neg.sum()
            losses[c] = loss
    else:
        losses = torch.zeros((num_classes,), dtype=ypred.dtype, device=ypred.device)
        for c in range(num_classes):
            y_neg = ypred[:,c,:,:]
            y_neg = y_neg[mask[:,c,:,:]<0.5]
            bce_neg = -torch.log(1-y_neg[c])
            if len(ypred_pos[c])>0:
                pred = torch.clamp(torch.cat(ypred_pos[c], dim=0), epsilon, 1-epsilon)
                bce_pos = -torch.log(pred)
                if loss_mode == 'ce_all':
                    loss = (bce_pos.sum()+bce_neg.sum())/(len(bce_pos)+len(bce_neg))
                elif loss_mode == 'ce_balance':
                    loss = (bce_pos.mean()+bce_neg.mean())/2
            else:
                loss = bce_neg.mean()
            losses[c] = loss

    return losses


def mil_approx_polar_sigmoid_loss(ypred, mask, gt_boxes_cr, 
        loss_mode='focal', weight_min=0.5, center_mode='fixed', 
        approx_method='softmax', approx_alpha=4, 
        pt_params={"output_shape": [90, 30], "scaling": "linear"}, 
        focal_params={'alpha':0.25, 'gamma':2.0}, 
        epsilon=1e-6):
    """ Compute the mil unary loss based on polar transformation.

    Args:
        ypred: Tensor of predicted data from the network with shape (B, C, W, H).
        mask:  Tensor of mask with shape (B, C, W, H), bounding box regions with value 1 and 0 otherwise.
        gt_boxes_cr: Tensor of boxes with (N, 5), where N is the number of bouding boxes in the batch,
                    the 5 elements of each row are [nb_img, class, center_x, center_r, radius]

    Returns:
        losses with shape (C,) for each category

    Reference:
        Wang, J. and Xia, B., 2022. Polar Transformation Based Multiple Instance 
        Learning Assisting Weakly Supervised Image Segmentation With Loose 
        Bounding Box Annotations. arXiv preprint arXiv:2203.06000.
    """
    assert loss_mode in ['ce_all', 'ce_balance', 'focal']
    assert approx_method in ['softmax', 'quasimax']
    assert center_mode in ['fixed', 'estimated']
    pt_params_used = copy.deepcopy(pt_params)
    ph, pw = pt_params['output_shape']
    assert (ph > 0) & ((pw == -1) | (pw > 0))
    dtype, device = ypred.dtype, ypred.device
    if pw > 0:
        d = pt_params["output_shape"][1]
        sigma2 = (d-1)**2 / (-2*math.log(weight_min))
        polar_weights = torch.exp(-torch.arange(d, dtype=dtype, device=device)**2/(2*sigma2))
        polar_weights = polar_weights.repeat(pt_params["output_shape"][0], 1)

    ypred = torch.clamp(ypred, epsilon, 1-epsilon)
    num_classes = ypred.shape[1]
    ob_img_index = gt_boxes_cr[:,0].type(torch.int32)
    ob_class_index = gt_boxes_cr[:,1].type(torch.int32)
    ob_gt_boxes_cr = gt_boxes_cr[:,2:]
    ypred_pos = {c:[] for c in range(num_classes)}
    for nb_ob in range(gt_boxes_cr.shape[0]):
        nb_img = ob_img_index[nb_ob]
        c = ob_class_index[nb_ob]
        cx,cy,r = ob_gt_boxes_cr[nb_ob,:].type(torch.int32)
        r = r + 5
        xmin = torch.clamp(cx-r,0)
        ymin = torch.clamp(cy-r,0)
        pred = ypred[nb_img,c,ymin:cy+r+1,xmin:cx+r+1]
        msk  = mask[nb_img,c,ymin:cy+r+1,xmin:cx+r+1]
        if pw == -1:
            pt_params_used['output_shape'] = [ph, r.item()]
            sigma2 = (pt_params_used['output_shape'][1]-1)**2 / (-2*math.log(weight_min))
            polar_weights = torch.exp(-torch.arange(pt_params_used['output_shape'][1], dtype=dtype, device=device)**2/(2*sigma2))
            polar_weights = polar_weights.repeat(ph, 1)

        if center_mode == 'fixed':
            p_center = torch.tensor((cy-ymin, cx-xmin), dtype=dtype, device=device)
        else:
            pred_msk = pred * (msk >= 0.5)
            cands = (pred_msk == torch.max(pred_msk)).nonzero()
            rand = torch.randperm(cands.shape[0])
            p_center = cands[rand[0], :]

        pred_polar = polar_transform(pred, center=p_center, radius=r, **pt_params_used)
        msk_polar = polar_transform(msk, center=p_center, radius=r, **pt_params_used)
        pred_polar = pred_polar * polar_weights
        
        msk_polar = msk_polar * (msk_polar > 0.5)
        flag = torch.sum(msk_polar,dim=1) > 0.5
        msk_polar = msk_polar[flag, :]
        pred_polar = pred_polar[flag, :]
        
        if approx_method == 'softmax':
            w = torch.exp(approx_alpha*pred_polar)
            prob = torch.sum(pred_polar*w*msk_polar,dim=1)/torch.sum(w*msk_polar,dim=1)
        elif approx_method == 'quasimax':
            w = torch.exp(approx_alpha*pred_polar)
            prob = torch.log(torch.sum(w*msk_polar,dim=1))/approx_alpha - torch.log(torch.sum(msk_polar, dim=1))/approx_alpha
        ypred_pos[c.item()].append(prob)
    
    if loss_mode == 'focal':
        alpha = focal_params['alpha']
        gamma = focal_params['gamma']
        losses = torch.zeros((num_classes,), dtype=ypred.dtype, device=ypred.device)
        for c in range(num_classes):
            y_neg = ypred[:,c,:,:]
            y_neg = y_neg[mask[:,c,:,:]<0.5]
            bce_neg = -(1-alpha)*(y_neg**gamma)*torch.log(1-y_neg)
            if len(ypred_pos[c])>0:
                y_pos = torch.cat(ypred_pos[c], dim=0)
                y_pos = torch.clamp(y_pos, epsilon, 1-epsilon)
                bce_pos = -alpha*((1-y_pos)**gamma)*torch.log(y_pos)
                loss = (bce_neg.sum()+bce_pos.sum())/len(bce_pos)
            else:
                loss = bce_neg.sum()
            losses[c] = loss
    else:
        losses = torch.zeros((num_classes,), dtype=ypred.dtype, device=ypred.device)
        for c in range(num_classes):
            y_neg = ypred[:,c,:,:]
            y_neg = y_neg[mask[:,c,:,:]<0.5]
            bce_neg = -torch.log(1-y_neg[c])
            if len(ypred_pos[c])>0:
                pred = torch.clamp(torch.cat(ypred_pos[c], dim=0), epsilon, 1-epsilon)
                bce_pos = -torch.log(pred)
                if loss_mode == 'ce_all':
                    loss = (bce_pos.sum()+bce_neg.sum())/(len(bce_pos)+len(bce_neg))
                elif loss_mode == 'ce_balance':
                    loss = (bce_pos.mean()+bce_neg.mean())/2
            else:
                loss = bce_neg.mean()
            losses[c] = loss
    return losses


def mil_unary_bbox_pos_generalized_neg_sigmoid_loss(ypred, mask, gt_boxes_xxyy,
        loss_mode='focal', focal_params={'alpha':0.25, 'gamma':2.0}, epsilon=1e-6):
    """ Compute the mil unary loss. A positive bag is defined as all pixels in a bounding box;
        generalized negative bag is used.

    Args:
        ypred: Tensor of predicted data from the network with shape (B, C, W, H).
        mask:  Tensor of mask with shape (B, C, W, H), bounding box regions with value 1 and 0 otherwise.
        gt_boxes_xxyy: Tensor of boxes with (N, 6), where N is the number of bouding boxes in the batch,
                    the 6 elements of each row are [nb_img, class, x1, y1, x2, y2]

    Returns:
        losses with shape (C,) for each category

    Reference:
        Wang, J. and Xia, B., 2021, September. Bounding box tightness prior
        for weakly supervised image segmentation. In International Conference on 
        Medical Image Computing and Computer-Assisted Intervention (pp. 526-536). 
        Springer, Cham.
    """
    assert loss_mode in ['ce_all', 'ce_balance', 'focal']
    ypred = torch.clamp(ypred, epsilon, 1-epsilon)
    num_classes = ypred.shape[1]
    ypred_pos = {c:[] for c in range(num_classes)}
    for nb_ob in range(gt_boxes_xxyy.shape[0]):
        nb_img = gt_boxes_xxyy[nb_ob,0]
        c = gt_boxes_xxyy[nb_ob,1].item()
        box = gt_boxes_xxyy[nb_ob,2:]
        pred = ypred[nb_img,c,box[1]:box[3]+1,box[0]:box[2]+1]
        if pred.numel() == 0:
            continue
        ypred_pos[c].append(pred.max())

    if loss_mode == 'focal':
        alpha = focal_params['alpha']
        gamma = focal_params['gamma']
        losses = torch.zeros((num_classes,), dtype=ypred.dtype, device=ypred.device)
        for c in range(num_classes):
            y_neg = ypred[:,c,:,:]
            y_neg = y_neg[mask[:,c,:,:]<0.5]
            bce_neg = -(1-alpha)*(y_neg**gamma)*torch.log(1-y_neg)
            if len(ypred_pos[c])>0:
                y_pos = torch.cat(ypred_pos[c], dim=0)
                y_pos = torch.clamp(y_pos, epsilon, 1-epsilon)
                bce_pos = -alpha*((1-y_pos)**gamma)*torch.log(y_pos)
                loss = (bce_neg.sum()+bce_pos.sum())/len(bce_pos)
            else:
                loss = bce_neg.sum()
            losses[c] = loss
    else:
        losses = torch.zeros((num_classes,), dtype=ypred.dtype, device=ypred.device)
        for c in range(num_classes):
            y_neg = ypred[:,c,:,:]
            y_neg = y_neg[mask[:,c,:,:]<0.5]
            bce_neg = -torch.log(1-y_neg[c])
            if len(ypred_pos[c])>0:
                bce_pos = -torch.log(torch.cat(ypred_pos[c], dim=0))
                if loss_mode == 'ce_all':
                    loss = (bce_pos.sum()+bce_neg.sum())/(len(bce_pos)+len(bce_neg))
                elif loss_mode == 'ce_balance':
                    loss = (bce_pos.mean()+bce_neg.mean())/2
            else:
                loss = bce_neg.mean()
            losses[c] = loss    
    return losses


def mil_approx_unary_bbox_pos_generalized_neg_sigmoid_loss(ypred, mask, gt_boxes_xxyy, 
        loss_mode='focal', approx_method='softmax', approx_alpha=4, 
        focal_params={'alpha':0.25, 'gamma':2.0}, epsilon=1e-6):
    """ Compute the mil unary loss. A positive bag is defined as all pixels in a bounding box;
        generalized negative bag is used.

    Args:
        ypred: Tensor of predicted data from the network with shape (B, C, W, H).
        mask:  Tensor of mask with shape (B, C, W, H), bounding box regions with value 1 and 0 otherwise.
        gt_boxes_xxyy: Tensor of boxes with (N, 6), where N is the number of bouding boxes in the batch,
                    the 6 elements of each row are [nb_img, class, x1, y1, x2, y2]

    Returns:
        losses with shape (C,) for each category

    Reference:
        Wang, J. and Xia, B., 2021, September. Bounding box tightness prior
        for weakly supervised image segmentation. In International Conference on 
        Medical Image Computing and Computer-Assisted Intervention (pp. 526-536). 
        Springer, Cham.
    """
    assert loss_mode in ['ce_all', 'ce_balance', 'focal']
    ypred = torch.clamp(ypred, epsilon, 1-epsilon)
    num_classes = ypred.shape[1]
    ypred_pos = {c:[] for c in range(num_classes)}
    for nb_ob in range(gt_boxes_xxyy.shape[0]):
        nb_img = gt_boxes_xxyy[nb_ob,0]
        c = gt_boxes_xxyy[nb_ob,1].item()
        box = gt_boxes_xxyy[nb_ob,2:]
        pred_org = ypred[nb_img,c,box[1]:box[3]+1,box[0]:box[2]+1]
        nb_pixels = pred_org.numel()
        if nb_pixels == 0:
            continue
        pred = torch.exp(approx_alpha*pred_org)
        if approx_method == 'softmax':
            prob = torch.sum(pred_org*pred)/pred.sum()
        elif approx_method == 'quasimax':
            n = pred_org.numel()
            prob = torch.log(pred.sum())/approx_alpha - math.log(nb_pixels)/approx_alpha
        ypred_pos[c].append(prob)

    if loss_mode == 'focal':
        alpha = focal_params['alpha']
        gamma = focal_params['gamma']
        losses = torch.zeros((num_classes,), dtype=ypred.dtype, device=ypred.device)
        for c in range(num_classes):
            y_neg = ypred[:,c,:,:]
            y_neg = y_neg[mask[:,c,:,:]<0.5]
            bce_neg = -(1-alpha)*(y_neg**gamma)*torch.log(1-y_neg)
            if len(ypred_pos[c])>0:
                y_pos = torch.stack(ypred_pos[c], dim=0)
                y_pos = torch.clamp(y_pos, epsilon, 1-epsilon)
                bce_pos = -alpha*((1-y_pos)**gamma)*torch.log(y_pos)
                loss = (bce_neg.sum()+bce_pos.sum())/len(bce_pos)
            else:
                loss = bce_neg.sum()
            losses[c] = loss
    else:
        losses = torch.zeros((num_classes,), dtype=ypred.dtype, device=ypred.device)
        for c in range(num_classes):
            y_neg = ypred[:,c,:,:]
            y_neg = y_neg[mask[:,c,:,:]<0.5]
            bce_neg = -torch.log(1-y_neg[c])
            if len(ypred_pos[c])>0:
                bce_pos = -torch.log(torch.stack(ypred_pos[c], dim=0))
                if loss_mode == 'ce_all':
                    loss = (bce_pos.sum()+bce_neg.sum())/(len(bce_pos)+len(bce_neg))
                elif loss_mode == 'ce_balance':
                    loss = (bce_pos.mean()+bce_neg.mean())/2
            else:
                loss = bce_neg.mean()
            losses[c] = loss    
    return losses


def mil_pairwise_loss(ypred, mask, softmax=True, exp_coef=-1):
    """ Compute the pair-wise loss.
        As defined in Weakly Supervised Instance Segmentation using the Bounding Box Tightness Prior
        
    Args:
        ypred: Tensor of predicted data from the network with shape (B, C, W, H).
        mask:  Tensor of mask with shape (B, C, W, H), bounding box regions with value 1 and 0 otherwise.

    Returns:
        pair-wise loss for each category (C,)
    """
    device = ypred.device
    center_weight = torch.tensor([[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]])
    pairwise_weights_list = [
            torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 0., 0.]]),  
            torch.tensor([[0., 0., 0.], [0., 0., 1.], [0., 0., 0.]]),  
            torch.tensor([[0., 1., 0.], [0., 0., 0.], [0., 0., 0.]]),  
            torch.tensor([[0., 0., 0.], [0., 0., 0.], [0., 1., 0.]]),  
            torch.tensor([[1., 0., 0.], [0., 0., 0.], [0., 0., 0.]]),  
            torch.tensor([[0., 0., 1.], [0., 0., 0.], [0., 0., 0.]]),  
            torch.tensor([[0., 0., 0.], [0., 0., 0.], [1., 0., 0.]]),  
            torch.tensor([[0., 0., 0.], [0., 0., 0.], [0., 0., 1.]])]
    ## pairwise loss for each col/row MIL
    num_classes = ypred.shape[1]
    if softmax:
        num_classes = num_classes - 1
    losses    = torch.zeros((num_classes,), dtype=ypred.dtype, device=device)
    for c in range(num_classes):
        pairwise_loss = []
        for w in pairwise_weights_list:
            weights = center_weight - w
            weights = weights.view(1, 1, 3, 3).to(device)
            aff_map = F.conv2d(ypred[:,c,:,:].unsqueeze(1), weights, padding=1)
            cur_loss = aff_map**2
            if exp_coef>0:
                cur_loss = torch.exp(exp_coef*cur_loss)-1
            cur_loss = torch.sum(cur_loss*mask[:,c,:,:].unsqueeze(1))/(torch.sum(mask[:,c,:,:]+1e-6))
            pairwise_loss.append(cur_loss)
        losses[c] = torch.mean(torch.stack(pairwise_loss))
    return losses
