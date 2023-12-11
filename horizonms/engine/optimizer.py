from torch import nn
import torch
from torch.optim import Optimizer
from collections import Iterable


def add_weight_decay(model, decay_value, skip_list=()):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad: 
            continue # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list: 
            no_decay.append(param)
        else: 
            decay.append(param)
    return [{'params': no_decay, 'weight_decay': 0.}, {'params': decay, 'weight_decay': decay_value}]


# Compute norm depending on the shape of x
def unitwise_norm(x):
    if (len(torch.squeeze(x).shape)) <= 1:  # Scalars, vectors
        axis = 0
        keepdims = False
    elif len(x.shape) in [2, 3]:  # Linear layers
        # Original code: IO
        # Pytorch: OI
        axis = 1
        keepdims = True
    elif len(x.shape) == 4:  # Conv kernels
        # Original code: HWIO
        # Pytorch: OIHW
        axis = [1, 2, 3]
        keepdims = True
    else:
        raise ValueError(f'Got a parameter with len(shape) not in [1, 2, 3, 4]! {x}')

    return torch.sqrt(torch.sum(torch.square(x), axis=axis, keepdim=keepdims))


class AGC(Optimizer):
    """Generic implementation of the Adaptive Gradient Clipping.

    Args:
      params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
      optim (torch.optim.Optimizer): Optimizer with base class optim.Optimizer
      clipping (float, optional): clipping value (default: 1e-3)
      eps (float, optional): eps (default: 1e-3)
      model (torch.nn.Module, optional): The original model
      ignore_agc (str, Iterable, optional): Layers for AGC to ignore
    """

    def __init__(self, params, optim: Optimizer, clipping: float = 1e-2, eps: float = 1e-3, model=None,
                 ignore_agc=["fc"]):

        if clipping < 0.0:
            raise ValueError("Invalid clipping value: {}".format(clipping))
        if eps < 0.0:
            raise ValueError("Invalid eps value: {}".format(eps))

        self.optim = optim

        defaults = dict(clipping=clipping, eps=eps)
        defaults = {**defaults, **optim.defaults}

        if not isinstance(ignore_agc, Iterable):
            ignore_agc = [ignore_agc]

        if model is not None:
            assert ignore_agc not in [
                None, []], "You must specify ignore_agc for AGC to ignore fc-like(or other) layers"
            names = [name for name, module in model.named_modules()]

            #print(names)

            for module_name in ignore_agc:
                if module_name not in names:
                    raise ModuleNotFoundError(
                        "Module name {} not found in the model".format(module_name))
            params = [{"params": list(module.parameters())} for name,
                                                                module in model.named_modules() if
                      name not in ignore_agc]

        else:
            params = [{"params": params}]

        self.agc_params = params
        self.eps = eps
        self.clipping = clipping

        self.defaults = defaults
        self.param_groups = optim.param_groups
        self.state = optim.state

        # super(AGC, self).__init__([], defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.agc_params:
            for p in group['params']:
                if p.grad is None:
                    continue

                param_norm = torch.max(unitwise_norm(
                    p.detach()), torch.tensor(self.eps,device=p.device))
                    # p.detach()), torch.tensor(self.eps).to(p.device))
                grad_norm = unitwise_norm(p.grad.detach())
                max_norm = param_norm * self.clipping

                trigger = grad_norm > max_norm

                # clipped_grad = p.grad *  (max_norm / torch.max(grad_norm,torch.tensor(1e-6).to(grad_norm.device)))
                clipped_grad = p.grad *  (max_norm / torch.max(grad_norm,torch.tensor(1e-6,device=grad_norm.device)))
                p.grad.detach().data.copy_(torch.where(trigger, clipped_grad, p.grad))

        return self.optim.step(closure)

    def zero_grad(self, set_to_none: bool = False):
        r"""Sets the gradients of all optimized :class:`torch.Tensor` s to zero.

        Args:
            set_to_none (bool): instead of setting to zero, set the grads to None.
                This is will in general have lower memory footprint, and can modestly improve performance.
                However, it changes certain behaviors. For example:
                1. When the user tries to access a gradient and perform manual ops on it,
                a None attribute or a Tensor full of 0s will behave differently.
                2. If the user requests ``zero_grad(set_to_none=True)`` followed by a backward pass, ``.grad``\ s
                are guaranteed to be None for params that did not receive a gradient.
                3. ``torch.optim`` optimizers have a different behavior if the gradient is 0 or None
                (in one case it does the step with a gradient of 0 and in the other it skips
                the step altogether).
        """
        for group in self.agc_params:
            for p in group['params']:
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        if p.grad.grad_fn is not None:
                            p.grad.detach_()
                        else:
                            p.grad.requires_grad_(False)
                        p.grad.zero_()


# This is a copy of the pytorch SGD implementation
# enhanced with gradient clipping
class SGD_AGC(Optimizer):
    def __init__(self, named_params, lr: float, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, clipping: float = None, eps: float = 1e-3):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov,
                        # Extra defaults
                        clipping=clipping,
                        eps=eps
                        )

        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")

        # Put params in list so each one gets its own group
        params = []
        for name, param in named_params:
            params.append({'params': param, 'name': name})

        super(SGD_AGC, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD_AGC, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            # Extra values for clipping
            clipping = group['clipping']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad

                # =========================
                # Gradient clipping
                if clipping is not None:
                    # param_norm = torch.maximum(unitwise_norm(p), torch.tensor(eps).to(p.device))
                    param_norm = torch.maximum(unitwise_norm(p), torch.tensor(eps,device=p.device))
                    grad_norm = unitwise_norm(d_p)
                    max_norm = param_norm * group['clipping']

                    trigger_mask = grad_norm > max_norm
                    # clipped_grad = p.grad * (max_norm / torch.maximum(grad_norm, torch.tensor(1e-6).to(p.device)))
                    clipped_grad = p.grad * (max_norm / torch.maximum(grad_norm, torch.tensor(1e-6,device=p.device)))
                    d_p = torch.where(trigger_mask, clipped_grad, d_p)
                # =========================

                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                p.add_(d_p, alpha=-group['lr'])

        return loss
