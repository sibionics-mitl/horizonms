from .bbox_sigmoid_func import mil_unary_baseline_sigmoid_loss,\
    mil_approx_unary_baseline_sigmoid_loss, \
    mil_unary_baseline_pos_generalized_neg_sigmoid_loss, \
    mil_approx_unary_baseline_pos_generalized_neg_sigmoid_loss, \
    mil_unary_parallel_sigmoid_loss, \
    mil_approx_unary_parallel_sigmoid_loss, \
    mil_approx_polar_sigmoid_loss, \
    mil_unary_bbox_pos_generalized_neg_sigmoid_loss, \
    mil_approx_unary_bbox_pos_generalized_neg_sigmoid_loss, \
    mil_pairwise_loss
from .. import LOSSES


__all__ = ["MILUnaryBaselineSigmoidLoss", "MILApproxUnaryBaselineSigmoidLoss",
           "MILUnaryBaselinePosGeneralizedNegSigmoidLoss",
           "MILApproxUnaryBaselinePosGeneralizedNegSigmoidLoss",
           "MILUnaryParallelSigmoidLoss", "MILApproxUnaryParallelSigmoidLoss",
           "MILApproxUnaryPolarSigmoidLoss",
           "MILUnaryBboxPosGeneralizedNegSigmoidLoss",
           "MILApproxUnaryBboxPosGeneralizedNegSigmoidLoss",
           "MILPairwiseLoss"]


@LOSSES.register_module()
class MILUnaryBaselineSigmoidLoss():
    r"""Compute the MIL baseline unary loss. 

    Args:
        loss_mode (str): type of loss. It is `'ce_all'` for cross entropy loss for all samples, 
            `'ce_balance'` for cross entropy loss for positive and negative classes , or `'focal'` for focal loss.
            Default: `'ce_all'`.
        focal_params (Dict): parameters for focal loss. Default: `{'alpha':0.25, 'gamma':2.0}`.
        epsilon (float): a small number for the stability of the loss calcualtion. Default: `1e-6`.

    Reference:
        Wang, J. and Xia, B., 2021, September. Bounding box tightness prior
        for weakly supervised image segmentation. In International Conference on 
        Medical Image Computing and Computer-Assisted Intervention (pp. 526-536). 
        Springer, Cham.
    """
    def __init__(self, loss_mode='ce_all', focal_params={'alpha':0.25, 'gamma':2.0},
                epsilon=1e-6):
        super(MILUnaryBaselineSigmoidLoss, self).__init__()
        self.loss_mode = loss_mode
        self.focal_params = focal_params
        self.epsilon = epsilon
        
    def __call__(self, ypred, gt_boxes_mask, gt_boxes_xxyy):
        r"""
        Args:
            ypred (Tensor): network prediction with shape (B, C, W, H).
            gt_boxes_mask (Tensor): binary mask with shape (B, C, W, H), in which bounding box regions have value 1, and 0 otherwise.
            gt_boxes_xxyy (Tensor): boxes with shape (N, 6), where N is the number of bouding boxes in the batch,
                the 6 elements of each row are `[nb_img, class, x1, y1, x2, y2]`, where `nb_img` is the index of image in the batch,
                `class` is the category of the object, `[x1, y1, x2, y2]` are left, up, right, down locations of bounding box.
        
        Returns:
            Tensor: loss values with shape (C,) for each category.
        """
        loss = mil_unary_baseline_sigmoid_loss(ypred, gt_boxes_mask, gt_boxes_xxyy,
            self.loss_mode, focal_params=self.focal_params, epsilon=self.epsilon)
        return loss


@LOSSES.register_module()
class MILApproxUnaryBaselineSigmoidLoss():
    r"""Compute the MIL baseline unary loss while applying smooth maximum function. 

    Args:
        loss_mode (str): type of loss. It is `'ce_all'` for cross entropy loss for all samples, 
            `'ce_balance'` for cross entropy loss for positive and negative classes , or `'focal'` for focal loss.
            Default: `'ce_all'`.
        approx_method (str): name of smooth maximum function. It is `'softmax'` or `'quasimax'`. Default: `'softmax'`.
        approx_alpha (float): parameter of smooth maximum function. Default: `4`.
        focal_params (Dict): parameters for focal loss. Default: `{'alpha':0.25, 'gamma':2.0}`.
        epsilon (float): a small number for the stability of the loss calcualtion. Default: `1e-6`.

    Reference:
        Wang, J. and Xia, B., 2021, September. Bounding box tightness prior
        for weakly supervised image segmentation. In International Conference on 
        Medical Image Computing and Computer-Assisted Intervention (pp. 526-536). 
        Springer, Cham.
    """
    def __init__(self, loss_mode='ce_all', approx_method='softmax', approx_alpha=4,
                focal_params={'alpha':0.25, 'gamma':2.0}, epsilon=1e-6):
        super(MILApproxUnaryBaselineSigmoidLoss, self).__init__()
        self.loss_mode = loss_mode
        self.approx_method = approx_method
        self.approx_alpha = approx_alpha
        self.focal_params = focal_params
        self.epsilon = epsilon
        
    def __call__(self, ypred, gt_boxes_mask, gt_boxes_xxyy):
        r"""
        Args:
            ypred (Tensor): network prediction with shape (B, C, W, H).
            gt_boxes_mask (Tensor): binary mask with shape (B, C, W, H), in which bounding box regions have value 1, and 0 otherwise.
            gt_boxes_xxyy (Tensor): boxes with shape (N, 6), where N is the number of bouding boxes in the batch,
                the 6 elements of each row are `[nb_img, class, x1, y1, x2, y2]`, where `nb_img` is the index of image in the batch,
                `class` is the category of the object, `[x1, y1, x2, y2]` are left, up, right, down locations of bounding box.
        
        Returns:
            Tensor: loss values with shape (C,) for each category.
        """
        loss = mil_approx_unary_baseline_sigmoid_loss(ypred, gt_boxes_mask, gt_boxes_xxyy,
            self.loss_mode,
            approx_method=self.approx_method, approx_alpha=self.approx_alpha, 
            focal_params=self.focal_params, epsilon=self.epsilon)
        return loss


@LOSSES.register_module()
class MILUnaryBaselinePosGeneralizedNegSigmoidLoss():
    r"""Compute the MIL baseline unary loss, in which the generalized negative bag is used.

    Args:
        loss_mode (str): type of loss. It is `'ce_all'` for cross entropy loss for all samples, 
            `'ce_balance'` for cross entropy loss for positive and negative classes , or `'focal'` for focal loss.
            Default: `'ce_all'`.
        focal_params (Dict): parameters for focal loss. Default: `{'alpha':0.25, 'gamma':2.0}`.
        epsilon (float): a small number for the stability of the loss calcualtion. Default: `1e-6`.

    Reference:
        Wang, J. and Xia, B., 2021, September. Bounding box tightness prior
        for weakly supervised image segmentation. In International Conference on 
        Medical Image Computing and Computer-Assisted Intervention (pp. 526-536). 
        Springer, Cham.
    """
    def __init__(self, loss_mode='ce_all', focal_params={'alpha':0.25, 'gamma':2.0},
                epsilon=1e-6):
        super(MILUnaryBaselinePosGeneralizedNegSigmoidLoss, self).__init__()
        self.loss_mode = loss_mode
        self.focal_params = focal_params
        self.epsilon = epsilon
        
    def __call__(self, ypred, gt_boxes_mask, gt_boxes_xxyy):
        r"""
        Args:
            ypred (Tensor): network prediction with shape (B, C, W, H).
            gt_boxes_mask (Tensor): binary mask with shape (B, C, W, H), in which bounding box regions have value 1, and 0 otherwise.
            gt_boxes_xxyy (Tensor): boxes with shape (N, 6), where N is the number of bouding boxes in the batch,
                the 6 elements of each row are `[nb_img, class, x1, y1, x2, y2]`, where `nb_img` is the index of image in the batch,
                `class` is the category of the object, `[x1, y1, x2, y2]` are left, up, right, down locations of bounding box.
        
        Returns:
            Tensor: loss values with shape (C,) for each category.
        """
        loss = mil_unary_baseline_pos_generalized_neg_sigmoid_loss(ypred, gt_boxes_mask,
            gt_boxes_xxyy, self.loss_mode, focal_params=self.focal_params,
            epsilon=self.epsilon)
        return loss


@LOSSES.register_module()
class MILApproxUnaryBaselinePosGeneralizedNegSigmoidLoss():
    r"""Compute the MIL baseline unary loss while applying smooth maximum function,
    in which the generalized negative bag is used.

    Args:
        loss_mode (str): type of loss. It is `'ce_all'` for cross entropy loss for all samples, 
            `'ce_balance'` for cross entropy loss for positive and negative classes , or `'focal'` for focal loss.
            Default: `'ce_all'`.
        approx_method (str): name of smooth maximum function. It is `'softmax'` or `'quasimax'`. Default: `'softmax'`.
        approx_alpha (float): parameter of smooth maximum function. Default: `4`.
        focal_params (Dict): parameters for focal loss. Default: `{'alpha':0.25, 'gamma':2.0}`.
        epsilon (float): a small number for the stability of the loss calcualtion. Default: `1e-6`.

    Reference:
        Wang, J. and Xia, B., 2021, September. Bounding box tightness prior
        for weakly supervised image segmentation. In International Conference on 
        Medical Image Computing and Computer-Assisted Intervention (pp. 526-536). 
        Springer, Cham.
    """
    def __init__(self, loss_mode='ce_all', approx_method='softmax', approx_alpha=4,
                focal_params={'alpha':0.25, 'gamma':2.0}, epsilon=1e-6):
        super(MILApproxUnaryBaselinePosGeneralizedNegSigmoidLoss, self).__init__()
        self.loss_mode = loss_mode
        self.approx_method = approx_method
        self.approx_alpha = approx_alpha
        self.focal_params = focal_params
        self.epsilon = epsilon
        
    def __call__(self, ypred, gt_boxes_mask, gt_boxes_xxyy):
        r"""
        Args:
            ypred (Tensor): network prediction with shape (B, C, W, H).
            gt_boxes_mask (Tensor): binary mask with shape (B, C, W, H), in which bounding box regions have value 1, and 0 otherwise.
            gt_boxes_xxyy (Tensor): boxes with shape (N, 6), where N is the number of bouding boxes in the batch,
                the 6 elements of each row are `[nb_img, class, x1, y1, x2, y2]`, where `nb_img` is the index of image in the batch,
                `class` is the category of the object, `[x1, y1, x2, y2]` are left, up, right, down locations of bounding box.
        
        Returns:
            Tensor: loss values with shape (C,) for each category.
        """
        loss = mil_approx_unary_baseline_pos_generalized_neg_sigmoid_loss(ypred, 
            gt_boxes_mask, gt_boxes_xxyy, self.loss_mode, approx_method=self.approx_method, 
            approx_alpha=self.approx_alpha, focal_params=self.focal_params, 
            epsilon=self.epsilon)
        return loss


@LOSSES.register_module()
class MILUnaryParallelSigmoidLoss():
    r"""Compute parallel transformation based MIL unary loss while no smooth maximum function is applied.

    Args:
        loss_mode (str): type of loss. It is `'ce_all'` for cross entropy loss for all samples, 
            `'ce_balance'` for cross entropy loss for positive and negative classes , or `'focal'` for focal loss.
            Default: `'ce_all'`.
        angle_params (tuple): degree of parallel crossing lines in the format of (start, stop, step). Default: `(-60,61,30)`.
        focal_params (Dict): parameters for focal loss. Default: `{'alpha':0.25, 'gamma':2.0}`.
        epsilon (float): a small number for the stability of the loss calcualtion. Default: `1e-6`.

    Reference:
        Wang, J. and Xia, B., 2021, September. Bounding box tightness prior
        for weakly supervised image segmentation. In International Conference on 
        Medical Image Computing and Computer-Assisted Intervention (pp. 526-536). 
        Springer, Cham.
        Wang, J. and Xia, B., 2023. Weakly Supervised Image Segmentation Beyond Tight Bounding Box Annotations. 
        arXiv preprint arXiv:2301.12053.
    """
    def __init__(self, loss_mode='ce_all', angle_params=(-60,61,30), 
                focal_params={'alpha':0.25, 'gamma':2.0}, epsilon=1e-6):
        super(MILUnaryParallelSigmoidLoss, self).__init__()
        self.loss_mode = loss_mode
        self.angle_params = angle_params
        self.focal_params = focal_params
        self.epsilon = epsilon
        
    def __call__(self, ypred, gt_boxes_mask, gt_boxes_cr):
        r"""
        Args:
            ypred (Tensor): network prediction with shape (B, C, W, H).
            gt_boxes_mask (Tensor): binary mask with shape (B, C, W, H), in which bounding box regions have value 1, and 0 otherwise.
            gt_boxes_cr (Tensor): boxes with (N, 5), where N is the number of bouding boxes in the batch,
                the 5 elements of each row are `[nb_img, class, center_x, center_y, radius]`, where `nb_img` is the index of image in the batch,
                `class` is the category of the object, `[center_x, center_y, radius]` are locations and radius of bounding box.

        Returns:
            loss values with shape (C,) for each category.
        """
        loss = mil_unary_parallel_sigmoid_loss(ypred, gt_boxes_mask, gt_boxes_cr,
            self.loss_mode, angle_params=self.angle_params,
            focal_params=self.focal_params, epsilon=self.epsilon)
        return loss


@LOSSES.register_module()
class MILApproxUnaryParallelSigmoidLoss():
    r"""Compute parallel transformation based MIL unary loss.

    Args:
        loss_mode (str): type of loss. It is `'ce_all'` for cross entropy loss for all samples, 
            `'ce_balance'` for cross entropy loss for positive and negative classes , or `'focal'` for focal loss.
            Default: `'focal'`.
        angle_params (tuple): degree of parallel crossing lines in the format of (start, stop, step). Default: `(-60,61,30)`.
        approx_method (str): name of smooth maximum function. It is `'softmax'` or `'quasimax'`. Default: `'softmax'`.
        approx_alpha (float): parameter of smooth maximum function. Default: `4`.
        focal_params (Dict): parameters for focal loss. Default: `{'alpha':0.25, 'gamma':2.0}`.
        epsilon (float): a small number for the stability of the loss calcualtion. Default: `1e-6`.

    Reference:
        Wang, J. and Xia, B., 2021, September. Bounding box tightness prior
        for weakly supervised image segmentation. In International Conference on 
        Medical Image Computing and Computer-Assisted Intervention (pp. 526-536). 
        Springer, Cham.
        Wang, J. and Xia, B., 2023. Weakly Supervised Image Segmentation Beyond Tight Bounding Box Annotations. 
        arXiv preprint arXiv:2301.12053.
    """
    def __init__(self, loss_mode='focal', angle_params=(-60,61,30), 
                approx_method='softmax', approx_alpha=4, 
                focal_params={'alpha':0.25, 'gamma':2.0}, epsilon=1e-6):
        super(MILApproxUnaryParallelSigmoidLoss, self).__init__()
        self.loss_mode = loss_mode
        self.angle_params = angle_params
        self.approx_method = approx_method
        self.approx_alpha = approx_alpha
        self.focal_params = focal_params
        self.epsilon = epsilon
        
    def __call__(self, ypred, gt_boxes_mask, gt_boxes_cr):
        r"""
        Args:
            ypred (Tensor): network prediction with shape (B, C, W, H).
            gt_boxes_mask (Tensor): binary mask with shape (B, C, W, H), in which bounding box regions have value 1, and 0 otherwise.
            gt_boxes_cr (Tensor): boxes with (N, 5), where N is the number of bouding boxes in the batch,
                the 5 elements of each row are `[nb_img, class, center_x, center_y, radius]`, where `nb_img` is the index of image in the batch,
                `class` is the category of the object, `[center_x, center_y, radius]` are locations and radius of bounding box.

        Returns:
            loss values with shape (C,) for each category.
        """
        loss = mil_approx_unary_parallel_sigmoid_loss(ypred, gt_boxes_mask, gt_boxes_cr,
            self.loss_mode, angle_params=self.angle_params,
            approx_method=self.approx_method, approx_alpha=self.approx_alpha,
            focal_params=self.focal_params, epsilon=self.epsilon)
        return loss


@LOSSES.register_module()
class MILApproxUnaryPolarSigmoidLoss():
    r"""Compute polar transformation based MIL unary loss.

    Args:
        loss_mode (str): type of loss. It is `'ce_all'` for cross entropy loss for all samples, 
            `'ce_balance'` for cross entropy loss for positive and negative classes , or `'focal'` for focal loss.
            Default: `'focal'`.
        weight_min (float): minimum weight of the samples in a polar line. Default: `0.5`.
        center_mode (str): center of the polar transformation. It is `'fixed'` or `'estimated'`. Default: `'fixed'`.
        approx_method (str): name of smooth maximum function. It is `'softmax'` or `'quasimax'`. Default: `'softmax'`.
        approx_alpha (float): parameter of smooth maximum function. Default: `4`.
        pt_params (Dict): parameters of polar transformation. Keys in the dictionary are `'output_shape'` and `'scaling'`.
            Default: `{"output_shape": [90, 30], "scaling": "linear"}`.
        focal_params (Dict): parameters for focal loss. Default: `{'alpha':0.25, 'gamma':2.0}`.
        epsilon (float): a small number for the stability of the loss calcualtion. Default: `1e-6`.

    Reference:
        Wang, J. and Xia, B., 2023. Weakly Supervised Image Segmentation Beyond Tight Bounding Box Annotations. 
        arXiv preprint arXiv:2301.12053.
    """
    def __init__(self, loss_mode='focal', weight_min=0.5, center_mode='fixed', 
        approx_method='softmax', approx_alpha=4, 
        pt_params={"output_shape": [90, 30], "scaling": "linear"},
        focal_params={'alpha':0.25, 'gamma':2.0},
        epsilon=1e-6):
        super(MILApproxUnaryPolarSigmoidLoss, self).__init__()
        self.loss_mode = loss_mode
        self.weight_min = weight_min
        self.center_mode = center_mode
        self.approx_method = approx_method
        self.approx_alpha = approx_alpha
        self.pt_params = pt_params
        self.focal_params = focal_params
        self.epsilon = epsilon
        
    def __call__(self, ypred, gt_boxes_mask, gt_boxes_cr):
        r"""
        Args:
            ypred (Tensor): network prediction with shape (B, C, W, H).
            gt_boxes_mask (Tensor): binary mask with shape (B, C, W, H), in which bounding box regions have value 1, and 0 otherwise.
            gt_boxes_cr (Tensor): boxes with (N, 5), where N is the number of bouding boxes in the batch,
                the 5 elements of each row are `[nb_img, class, center_x, center_y, radius]`, where `nb_img` is the index of image in the batch,
                `class` is the category of the object, `[center_x, center_y, radius]` are locations and radius of bounding box.

        Returns:
            loss values with shape (C,) for each category.
        """
        loss = mil_approx_polar_sigmoid_loss(ypred, gt_boxes_mask, gt_boxes_cr, 
                self.loss_mode, weight_min=self.weight_min, center_mode=self.center_mode, 
                approx_method=self.approx_method, approx_alpha=self.approx_alpha,
                pt_params=self.pt_params, focal_params=self.focal_params,
                epsilon=self.epsilon)
        return loss



@LOSSES.register_module()
class MILUnaryBboxPosGeneralizedNegSigmoidLoss():
    r"""Compute MIL unary loss. A positive bag is defined as all pixels in a bounding box; generalized negative bag is used.

    Args:
        loss_mode (str): type of loss. It is `'ce_all'` for cross entropy loss for all samples, 
            `'ce_balance'` for cross entropy loss for positive and negative classes , or `'focal'` for focal loss.
            Default: `'focal'`.
        focal_params (Dict): parameters for focal loss. Default: `{'alpha':0.25, 'gamma':2.0}`.
        epsilon (float): a small number for the stability of the loss calcualtion. Default: `1e-6`.

    Reference:
        Wang, J. and Xia, B., 2021, September. Bounding box tightness prior
        for weakly supervised image segmentation. In International Conference on 
        Medical Image Computing and Computer-Assisted Intervention (pp. 526-536). 
        Springer, Cham.
    """
    def __init__(self, loss_mode='focal', focal_params={'alpha':0.25, 'gamma':2.0}, epsilon=1e-6):
        super(MILUnaryBboxPosGeneralizedNegSigmoidLoss, self).__init__()
        self.loss_mode = loss_mode
        self.focal_params = focal_params
        self.epsilon = epsilon
        
    def __call__(self, ypred, gt_boxes_mask, gt_boxes_xxyy):
        r"""
        Args:
            ypred (Tensor): network prediction with shape (B, C, W, H).
            gt_boxes_mask (Tensor): binary mask with shape (B, C, W, H), in which bounding box regions have value 1, and 0 otherwise.
            gt_boxes_xxyy (Tensor): boxes with shape (N, 6), where N is the number of bouding boxes in the batch,
                the 6 elements of each row are `[nb_img, class, x1, y1, x2, y2]`, where `nb_img` is the index of image in the batch,
                `class` is the category of the object, `[x1, y1, x2, y2]` are left, up, right, down locations of bounding box.
        
        Returns:
            Tensor: loss values with shape (C,) for each category.
        """
        loss = mil_unary_bbox_pos_generalized_neg_sigmoid_loss(ypred, 
            gt_boxes_mask, gt_boxes_xxyy, self.loss_mode, 
            focal_params=self.focal_params, epsilon=self.epsilon)
        return loss


@LOSSES.register_module()
class MILApproxUnaryBboxPosGeneralizedNegSigmoidLoss():
    r"""Compute MIL unary loss. A positive bag is defined as all pixels in a bounding box; generalized negative bag is used.

    Args:
        loss_mode (str): type of loss. It is `'ce_all'` for cross entropy loss for all samples, 
            `'ce_balance'` for cross entropy loss for positive and negative classes , or `'focal'` for focal loss.
            Default: `'focal'`.
        approx_method (str): name of smooth maximum function. It is `'softmax'` or `'quasimax'`. Default: `'softmax'`.
        approx_alpha (float): parameter of smooth maximum function. Default: `4`.
        focal_params (Dict): parameters for focal loss. Default: `{'alpha':0.25, 'gamma':2.0}`.
        epsilon (float): a small number for the stability of the loss calcualtion. Default: `1e-6`.

    Reference:
        Wang, J. and Xia, B., 2021, September. Bounding box tightness prior
        for weakly supervised image segmentation. In International Conference on 
        Medical Image Computing and Computer-Assisted Intervention (pp. 526-536). 
        Springer, Cham.
    """
    def __init__(self, loss_mode='focal', approx_method='softmax', approx_alpha=4,
                focal_params={'alpha':0.25, 'gamma':2.0}, epsilon=1e-6):
        super(MILApproxUnaryBboxPosGeneralizedNegSigmoidLoss, self).__init__()
        self.loss_mode = loss_mode
        self.approx_method = approx_method
        self.approx_alpha = approx_alpha
        self.focal_params = focal_params
        self.epsilon = epsilon
        
    def __call__(self, ypred, gt_boxes_mask, gt_boxes_xxyy):
        r"""
        Args:
            ypred (Tensor): network prediction with shape (B, C, W, H).
            gt_boxes_mask (Tensor): binary mask with shape (B, C, W, H), in which bounding box regions have value 1, and 0 otherwise.
            gt_boxes_xxyy (Tensor): boxes with shape (N, 6), where N is the number of bouding boxes in the batch,
                the 6 elements of each row are `[nb_img, class, x1, y1, x2, y2]`, where `nb_img` is the index of image in the batch,
                `class` is the category of the object, `[x1, y1, x2, y2]` are left, up, right, down locations of bounding box.
        
        Returns:
            Tensor: loss values with shape (C,) for each category.
        """
        loss = mil_approx_unary_bbox_pos_generalized_neg_sigmoid_loss(ypred, 
            gt_boxes_mask, gt_boxes_xxyy, self.loss_mode, 
            approx_method=self.approx_method, approx_alpha=self.approx_alpha, 
            focal_params=self.focal_params, epsilon=self.epsilon)
        return loss


@LOSSES.register_module()
class MILPairwiseLoss():
    r"""Compute pair-wise loss, defined in "Weakly Supervised Instance Segmentation using the Bounding Box Tightness Prior".
        
    Args:
        softmax (bool): If True, the network outputs softmax probability. Default: `True`.
        exp_coef (float): coefficient of exponential function applied to network output. It should be positive. 
            -1 denotes coefficient of exponential function is not applied. Default: `-1`.
    """
    def __init__(self, softmax=True, exp_coef=-1):
        super(MILPairwiseLoss, self).__init__()
        self.softmax = softmax
        self.exp_coef = exp_coef
        
    def __call__(self, ypred, gt_boxes_mask):
        r"""
        Args:
            ypred (Tensor): network prediction with shape (B, C, W, H).
            gt_boxes_mask (Tensor): binary mask with shape (B, C, W, H), in which bounding box regions have value 1, and 0 otherwise.
            
        Returns:
            Tensor: loss values with shape (C,) for each category.
        """
        loss = mil_pairwise_loss(ypred, gt_boxes_mask, softmax=self.softmax, exp_coef=self.exp_coef)
        return loss
