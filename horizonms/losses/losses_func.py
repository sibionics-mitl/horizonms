import torch
import torch.nn.functional as F


def smooth_l1_loss(target, preds, weight, sigma: float = 3, size_average: bool = True):
    """ Compute the smooth L1 loss of y_pred w.r.t. y_true.

    Args:
        y_true: Tensor from the generator of shape (N, 4). 
        y_pred: Tensor from the network of shape (N, 4).
        sigma: This argument defines the point where the loss changes from L2 to L1.

    Returns:
        The smooth L1 loss of y_pred w.r.t. y_true.
    """
    assert target.shape[-1]==preds.shape[-1]
    sigma_squared = sigma ** 2
        
    # compute smooth L1 loss
    # f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
    #        |x| - 0.5 / sigma / sigma    otherwise
    diff = torch.abs(preds - target)
    regression_loss = torch.where(diff<1.0/sigma_squared,
                                  0.5*sigma_squared*torch.pow(diff, 2),
                                  diff-0.5/sigma_squared
                                  )
    if weight is None:
        regression_loss = regression_loss.sum()
        normalizer = max(1, target.shape[0])
    else:
        weight = weight.reshape(-1, 1)
        regression_loss = (regression_loss * weight).sum()
        normalizer = weight.sum() + 1e-6

    if size_average:
        return regression_loss/normalizer
    else:
        return regression_loss


def weak_cross_entropy(ypred, mask, gt_boxes, mode: str = 'all', w_alpha: float = 4.0,
        toppk: float = -1.0, focal_params: dict = {'alpha':0.25, 'gamma':2.0},
        epsilon: float = 1e-10):
    ypred =  torch.clamp(ypred, epsilon, 1-epsilon)
    num_classes = ypred.shape[1]

    bw_pos = mask>=0.5
    if toppk > 0:
        bw_pos = mask.new_zeros(mask.shape, dtype=torch.bool)
        for nb_ob in range(gt_boxes.shape[0]):
            nb_img = gt_boxes[nb_ob,0]
            c      = gt_boxes[nb_ob,1].item()
            box    = gt_boxes[nb_ob,2:]
            pred   = ypred[nb_img,c,box[1]:box[3]+1,box[0]:box[2]+1]
            loc = int(pred.numel()*toppk)
            sorted, _ = torch.sort(pred.flatten(), descending=True)
            T = sorted[loc]
            bw_pos[nb_img, c, box[1]:box[3]+1, box[0]:box[2]+1] = (pred>T)
    num_pos = bw_pos.sum(dim=(0,2,3))

    if mode=='focal':
        alpha = focal_params['alpha']
        gamma = focal_params['gamma']
        loss_neg = -((mask<0.5)*(1-alpha)*(ypred**gamma)*torch.log(1-ypred)).sum(dim=(0,2,3))
        loss_pos = -(bw_pos*alpha*((1-ypred)**gamma)*torch.exp(w_alpha*(ypred-0.5))*torch.log(ypred)).sum(dim=(0,2,3))
        losses = (loss_neg + loss_pos)/num_pos
    else:
        num_neg = (mask<0.5).sum(dim=(0,2,3))
        loss_neg = -((mask<0.5)*torch.log(1-ypred)).sum(dim=(0,2,3))
        loss_pos = -(bw_pos*torch.exp(w_alpha*(ypred-0.5))*torch.log(ypred)).sum(dim=(0,2,3))
        loss_pos_weight = torch.sum(bw_pos*torch.exp(w_alpha*(ypred-0.5)), dim=(0,2,3))
        loss_pos = loss_pos / loss_pos_weight * num_pos
        if num_pos>0:
            if mode == 'all':
                losses = (loss_neg + loss_pos) / (num_neg + num_pos)
            elif mode=='balance':
                losses = loss_neg/num_neg + loss_pos/num_pos
        else:
            losses = loss_neg/num_neg
    return losses


def size_constraint_loss(ypred, ytrue, is_true: bool = True):
    size_pred = torch.sum(ypred, dim=(0,2,3))
    size_true = torch.sum(ytrue, dim=(0,2,3))  
    losses = (size_pred/size_true-1)**2
    if is_true==False: ## weakly supervised size using bounding box, and ytrue is mask
        flag = size_pred<size_true
        losses[flag] = 0
    # print('-------',size_pred, size_true, losses)
    return losses


# mil locations: pos
# threshold can be learnt from network using a new branch with num_classes outputs
def weak_bce_sigmoid_loss(ypred, gt_boxes, threshold=0.5, epsilon=1e-6):
    ypred =  torch.clamp(ypred, epsilon, 1-epsilon)
    num_classes = ypred.shape[1]
    # ypred_pos = {c:[] for c in range(num_classes)}
    ypred_box = {c:[] for c in range(num_classes)}
    label_box = {c:[] for c in range(num_classes)}
    for nb_ob in range(gt_boxes.shape[0]):
        nb_img = gt_boxes[nb_ob,0]
        c      = gt_boxes[nb_ob,1].item()
        box    = gt_boxes[nb_ob,2:]
        pred   = ypred[nb_img,c,box[1]:box[3]+1,box[0]:box[2]+1]
        label  = (pred>threshold)#.detach()
        # assign gt
        max_v0 = torch.max(pred,dim=0,keepdims=True)[0]
        max_v1 = torch.max(pred,dim=1,keepdims=True)[0]
        mil = ((pred==max_v0)|(pred==max_v1))#.detach()
        label[mil] = True
        label = label.float()
        ypred_box[c].append(torch.flatten(pred))
        label_box[c].append(torch.flatten(label))
        
    losses = torch.zeros((num_classes,), dtype=ypred.dtype, device=ypred.device)
    for c in range(num_classes):
        pred = torch.cat(ypred_box[c], dim=0)
        gt   = torch.cat(label_box[c], dim=0)
        loss_neg = -torch.mean((1-gt)*torch.log(1-pred))
        loss_pos = -torch.mean(gt*torch.log(pred))
        losses[c] = (loss_neg+loss_pos)/2.0
    return losses


def bbox_overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False, eps=1e-6):
    """Calculate overlap between two set of bboxes.

    If ``is_aligned `` is ``False``, then calculate the overlaps between each
    bbox of bboxes1 and bboxes2, otherwise the overlaps between each aligned
    pair of bboxes1 and bboxes2.

    Args:
        bboxes1 (Tensor): shape (B, m, 4) in <x1, y1, x2, y2> format or empty.
        bboxes2 (Tensor): shape (B, n, 4) in <x1, y1, x2, y2> format or empty.
            B indicates the batch dim, in shape (B1, B2, ..., Bn).
            If ``is_aligned `` is ``True``, then m and n must be equal.
        mode (str): "iou" (intersection over union), "iof" (intersection over
            foreground) or "giou" (generalized intersection over union).
            Default "iou".
        is_aligned (bool, optional): If True, then m and n must be equal.
            Default False.
        eps (float, optional): A value added to the denominator for numerical
            stability. Default 1e-6.

    Returns:
        Tensor: shape (m, n) if ``is_aligned `` is False else shape (m,)

    Example:
        >>> bboxes1 = torch.FloatTensor([
        >>>     [0, 0, 10, 10],
        >>>     [10, 10, 20, 20],
        >>>     [32, 32, 38, 42],
        >>> ])
        >>> bboxes2 = torch.FloatTensor([
        >>>     [0, 0, 10, 20],
        >>>     [0, 10, 10, 19],
        >>>     [10, 10, 20, 20],
        >>> ])
        >>> overlaps = bbox_overlaps(bboxes1, bboxes2)
        >>> assert overlaps.shape == (3, 3)
        >>> overlaps = bbox_overlaps(bboxes1, bboxes2, is_aligned=True)
        >>> assert overlaps.shape == (3, )

    Example:
        >>> empty = torch.empty(0, 4)
        >>> nonempty = torch.FloatTensor([[0, 0, 10, 9]])
        >>> assert tuple(bbox_overlaps(empty, nonempty).shape) == (0, 1)
        >>> assert tuple(bbox_overlaps(nonempty, empty).shape) == (1, 0)
        >>> assert tuple(bbox_overlaps(empty, empty).shape) == (0, 0)
    """

    assert mode in ['iou', 'iof', 'giou'], f'Unsupported mode {mode}'
    # Either the boxes are empty or the length of boxes's last dimenstion is 4
    assert (bboxes1.size(-1) == 4 or bboxes1.size(0) == 0)
    assert (bboxes2.size(-1) == 4 or bboxes2.size(0) == 0)

    # Batch dim must be the same
    # Batch dim: (B1, B2, ... Bn)
    assert bboxes1.shape[:-2] == bboxes2.shape[:-2]
    batch_shape = bboxes1.shape[:-2]

    rows = bboxes1.size(-2)
    cols = bboxes2.size(-2)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        if is_aligned:
            return bboxes1.new(batch_shape + (rows, ))
        else:
            return bboxes1.new(batch_shape + (rows, cols))

    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (
        bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (
        bboxes2[..., 3] - bboxes2[..., 1])

    if is_aligned:
        lt = torch.max(bboxes1[..., :2], bboxes2[..., :2])  # [B, rows, 2]
        rb = torch.min(bboxes1[..., 2:], bboxes2[..., 2:])  # [B, rows, 2]

        wh = (rb - lt).clamp(min=0)  # [B, rows, 2]
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ['iou', 'giou']:
            union = area1 + area2 - overlap
        else:
            union = area1
        if mode == 'giou':
            enclosed_lt = torch.min(bboxes1[..., :2], bboxes2[..., :2])
            enclosed_rb = torch.max(bboxes1[..., 2:], bboxes2[..., 2:])
    else:
        lt = torch.max(bboxes1[..., :, None, :2],
                       bboxes2[..., None, :, :2])  # [B, rows, cols, 2]
        rb = torch.min(bboxes1[..., :, None, 2:],
                       bboxes2[..., None, :, 2:])  # [B, rows, cols, 2]

        wh = (rb - lt).clamp(min=0)  # [B, rows, cols, 2]
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ['iou', 'giou']:
            union = area1[..., None] + area2[..., None, :] - overlap
        else:
            union = area1[..., None]
        if mode == 'giou':
            enclosed_lt = torch.min(bboxes1[..., :, None, :2],
                                    bboxes2[..., None, :, :2])
            enclosed_rb = torch.max(bboxes1[..., :, None, 2:],
                                    bboxes2[..., None, :, 2:])

    eps = union.new_tensor([eps])
    union = torch.max(union, eps)
    ious = overlap / union
    if mode in ['iou', 'iof']:
        return ious
    # calculate gious
    enclose_wh = (enclosed_rb - enclosed_lt).clamp(min=0)
    enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
    enclose_area = torch.max(enclose_area, eps)
    gious = ious - (enclose_area - union) / enclose_area
    return gious
