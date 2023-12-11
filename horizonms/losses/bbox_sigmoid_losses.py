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


__all__ = ("MILUnaryBaselineSigmoidLoss", "MILApproxUnaryBaselineSigmoidLoss",
           "MILUnaryBaselinePosGeneralizedNegSigmoidLoss",
           "MILApproxUnaryBaselinePosGeneralizedNegSigmoidLoss",
           "MILUnaryParallelSigmoidLoss", "MILApproxUnaryParallelSigmoidLoss",
           "MILApproxUnaryPolarSigmoidLoss",
           "MILUnaryBboxPosGeneralizedNegSigmoidLoss",
           "MILApproxUnaryBboxPosGeneralizedNegSigmoidLoss",
           "MILPairwiseLoss")


@LOSSES.register_module()
class MILUnaryBaselineSigmoidLoss():
    """ Compute the mil baseline unary loss. 

    Args:
        loss_mode (str): type of loss. It is `'ce_all'` for cross entropy loss for all samples, 
            `'ce_balance'` for cross entropy loss for positive and negative classes , or `'focal'` for focal loss.
        focal_params (Dict): parameters for focal loss.
        epsilon (float): a small number for the stability of the loss calcualtion.

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
        loss = mil_unary_baseline_sigmoid_loss(ypred, gt_boxes_mask, gt_boxes_xxyy,
            self.loss_mode, focal_params=self.focal_params, epsilon=self.epsilon)
        return loss


@LOSSES.register_module()
class MILApproxUnaryBaselineSigmoidLoss():
    """ Compute the mil baseline unary loss while applying smooth maximum function. 

    Args:
        loss_mode (str): type of loss. It is `'ce_all'` for cross entropy loss for all samples, 
            `'ce_balance'` for cross entropy loss for positive and negative classes , or `'focal'` for focal loss.
        approx_method (str): name of smooth maximum function. It is `'softmax'` or `'quasimax'`.
        approx_alpha (float): parameter of smooth maximum function.
        focal_params (Dict): parameters for focal loss.
        epsilon (float): a small number for the stability of the loss calcualtion.

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
        loss = mil_approx_unary_baseline_sigmoid_loss(ypred, gt_boxes_mask, gt_boxes_xxyy,
            self.loss_mode,
            approx_method=self.approx_method, approx_alpha=self.approx_alpha, 
            focal_params=self.focal_params, epsilon=self.epsilon)
        return loss


@LOSSES.register_module()
class MILUnaryBaselinePosGeneralizedNegSigmoidLoss():
    """ Compute the mil baseline unary loss, in which the generalized negative bag is used.

    Args:
        loss_mode (str): type of loss. It is `'ce_all'` for cross entropy loss for all samples, 
            `'ce_balance'` for cross entropy loss for positive and negative classes , or `'focal'` for focal loss.
        focal_params (Dict): parameters for focal loss.
        epsilon (float): a small number for the stability of the loss calcualtion.

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
        loss = mil_unary_baseline_pos_generalized_neg_sigmoid_loss(ypred, gt_boxes_mask,
            gt_boxes_xxyy, self.loss_mode, focal_params=self.focal_params,
            epsilon=self.epsilon)
        return loss


@LOSSES.register_module()
class MILApproxUnaryBaselinePosGeneralizedNegSigmoidLoss():
    """ Compute the mil baseline unary loss while applying smooth maximum function,
        in which the generalized negative bag is used.

    Args:
        loss_mode (str): type of loss. It is `'ce_all'` for cross entropy loss for all samples, 
            `'ce_balance'` for cross entropy loss for positive and negative classes , or `'focal'` for focal loss.
        approx_method (str): name of smooth maximum function. It is `'softmax'` or `'quasimax'`.
        approx_alpha (float): parameter of smooth maximum function.
        focal_params (Dict): parameters for focal loss.
        epsilon (float): a small number for the stability of the loss calcualtion.

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
        loss = mil_approx_unary_baseline_pos_generalized_neg_sigmoid_loss(ypred, 
            gt_boxes_mask, gt_boxes_xxyy, self.loss_mode, approx_method=self.approx_method, 
            approx_alpha=self.approx_alpha, focal_params=self.focal_params, 
            epsilon=self.epsilon)
        return loss


@LOSSES.register_module()
class MILUnaryParallelSigmoidLoss():
    """ Compute the mil generalized unary loss while no smooth maximum function is applied.

    Args:
        loss_mode (str): type of loss. It is `'ce_all'` for cross entropy loss for all samples, 
            `'ce_balance'` for cross entropy loss for positive and negative classes , or `'focal'` for focal loss.
        angle_params (tuple): degree of parallel crossing lines in the format of (start, stop, step).
        focal_params (Dict): parameters for focal loss.
        epsilon (float): a small number for the stability of the loss calcualtion.

    Reference:
        Wang, J. and Xia, B., 2021, September. Bounding box tightness prior
        for weakly supervised image segmentation. In International Conference on 
        Medical Image Computing and Computer-Assisted Intervention (pp. 526-536). 
        Springer, Cham.
    """
    def __init__(self, loss_mode='all', angle_params=(-60,61,30), 
                focal_params={'alpha':0.25, 'gamma':2.0}, epsilon=1e-6):
        super(MILUnaryParallelSigmoidLoss, self).__init__()
        self.loss_mode = loss_mode
        self.angle_params = angle_params
        self.focal_params = focal_params
        self.epsilon = epsilon
        
    def __call__(self, ypred, gt_boxes_mask, gt_boxes_cr):
        loss = mil_unary_parallel_sigmoid_loss(ypred, gt_boxes_mask, gt_boxes_cr,
            self.loss_mode, angle_params=self.angle_params,
            focal_params=self.focal_params, epsilon=self.epsilon)
        return loss


@LOSSES.register_module()
class MILApproxUnaryParallelSigmoidLoss():
    """ Compute the mil generalized unary loss.

    Args:
        loss_mode (str): type of loss. It is `'ce_all'` for cross entropy loss for all samples, 
            `'ce_balance'` for cross entropy loss for positive and negative classes , or `'focal'` for focal loss.
        angle_params (tuple): degree of parallel crossing lines in the format of (start, stop, step).
        approx_method (str): name of smooth maximum function. It is `'softmax'` or `'quasimax'`.
        approx_alpha (float): parameter of smooth maximum function.
        focal_params (Dict): parameters for focal loss.
        epsilon (float): a small number for the stability of the loss calcualtion.

    Reference:
        Wang, J. and Xia, B., 2021, September. Bounding box tightness prior
        for weakly supervised image segmentation. In International Conference on 
        Medical Image Computing and Computer-Assisted Intervention (pp. 526-536). 
        Springer, Cham.
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
        loss = mil_approx_unary_parallel_sigmoid_loss(ypred, gt_boxes_mask, gt_boxes_cr,
            self.loss_mode, angle_params=self.angle_params,
            approx_method=self.approx_method, approx_alpha=self.approx_alpha,
            focal_params=self.focal_params, epsilon=self.epsilon)
        return loss


@LOSSES.register_module()
class MILApproxUnaryPolarSigmoidLoss():
    """ Compute the mil unary loss based on polar transformation.

    Args:
        loss_mode (str): type of loss. It is `'ce_all'` for cross entropy loss for all samples, 
            `'ce_balance'` for cross entropy loss for positive and negative classes , or `'focal'` for focal loss.
        weight_min (float): minimum weight of the samples in a polar line.
        center_mode (str): center of the polar transformation. It is `'fixed'` or `'estimated'`.
        approx_method (str): name of smooth maximum function. It is `'softmax'` or `'quasimax'`.
        approx_alpha (float): parameter of smooth maximum function.
        pt_params (Dict): parameters of polar transformation. Keys in the dictionary are `'output_shape'` and `'scaling'`.
        focal_params (Dict): parameters for focal loss.
        epsilon (float): a small number for the stability of the loss calcualtion.

    Reference:
        Wang, J. and Xia, B., 2022. Polar Transformation Based Multiple Instance 
        Learning Assisting Weakly Supervised Image Segmentation With Loose 
        Bounding Box Annotations. arXiv preprint arXiv:2203.06000.
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
        loss = mil_approx_polar_sigmoid_loss(ypred, gt_boxes_mask, gt_boxes_cr, 
                self.loss_mode, weight_min=self.weight_min, center_mode=self.center_mode, 
                approx_method=self.approx_method, approx_alpha=self.approx_alpha,
                pt_params=self.pt_params, focal_params=self.focal_params,
                epsilon=self.epsilon)
        return loss



@LOSSES.register_module()
class MILUnaryBboxPosGeneralizedNegSigmoidLoss():
    """ Compute the mil unary loss. A positive bag is defined as all pixels in a bounding box;
        generalized negative bag is used.

    Args:
        loss_mode (str): type of loss. It is `'ce_all'` for cross entropy loss for all samples, 
            `'ce_balance'` for cross entropy loss for positive and negative classes , or `'focal'` for focal loss.
        focal_params (Dict): parameters for focal loss.
        epsilon (float): a small number for the stability of the loss calcualtion.

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
        loss = mil_unary_bbox_pos_generalized_neg_sigmoid_loss(ypred, 
            gt_boxes_mask, gt_boxes_xxyy, self.loss_mode, 
            focal_params=self.focal_params, epsilon=self.epsilon)
        return loss


@LOSSES.register_module()
class MILApproxUnaryBboxPosGeneralizedNegSigmoidLoss():
    """ Compute the mil unary loss. A positive bag is defined as all pixels in a bounding box;
        generalized negative bag is used.

    Args:
        loss_mode (str): type of loss. It is `'ce_all'` for cross entropy loss for all samples, 
            `'ce_balance'` for cross entropy loss for positive and negative classes , or `'focal'` for focal loss.
        approx_method (str): name of smooth maximum function. It is `'softmax'` or `'quasimax'`.
        approx_alpha (float): parameter of smooth maximum function.
        focal_params (Dict): parameters for focal loss.
        epsilon (float): a small number for the stability of the loss calcualtion.

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
        loss = mil_approx_unary_bbox_pos_generalized_neg_sigmoid_loss(ypred, 
            gt_boxes_mask, gt_boxes_xxyy, self.loss_mode, 
            approx_method=self.approx_method, approx_alpha=self.approx_alpha, 
            focal_params=self.focal_params, epsilon=self.epsilon)
        return loss


@LOSSES.register_module()
class MILPairwiseLoss():
    """ Compute the pair-wise loss.
        As defined in Weakly Supervised Instance Segmentation using the Bounding Box Tightness Prior.
        
    Args:
        softmax (bool): If True, the network outputs softmax probability.
        exp_coef (float): coefficient of exponential function applied to network output> It should be positive. 
            -1 denotes coefficient of exponential function is not applied.  Default: -1.
    """
    def __init__(self, softmax=True, exp_coef=-1):
        super(MILPairwiseLoss, self).__init__()
        self.softmax = softmax
        self.exp_coef = exp_coef
        
    def __call__(self, ypred, gt_boxes_mask):
        loss = mil_pairwise_loss(ypred, gt_boxes_mask, softmax=self.softmax, exp_coef=self.exp_coef)
        return loss
