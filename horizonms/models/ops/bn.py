from typing import Optional
import torch
from torch import Tensor
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.nn import Module, init


class BatchNormBase(Module):
    _version = 2
    __constants__ = ["track_running_stats", "momentum", "eps", "num_features", "affine"]
    num_features: int
    eps: float
    momentum: float
    affine: bool
    track_running_stats: bool
    # WARNING: weight and bias purposely not defined here.
    # See https://github.com/pytorch/pytorch/issues/39670
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device=None,
        dtype=None
    ):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(BatchNormBase, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = Parameter(torch.empty(num_features, **factory_kwargs))
            self.bias = Parameter(torch.empty(num_features, **factory_kwargs))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features, **factory_kwargs))
            self.register_buffer('running_var', torch.ones(num_features, **factory_kwargs))
            self.running_mean: Optional[Tensor]
            self.running_var: Optional[Tensor]
            self.register_buffer('num_batches_tracked',
                                 torch.tensor(0, dtype=torch.long,
                                              **{k: v for k, v in factory_kwargs.items() if k != 'dtype'}))
            self.num_batches_tracked: Optional[Tensor]
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)
            self.register_buffer("num_batches_tracked", None)
        self.reset_parameters()

    def reset_running_stats(self) -> None:
        if self.track_running_stats:
            # running_mean/running_var/num_batches... are registered at runtime depending
            # if self.track_running_stats is on
            self.running_mean.zero_()  # type: ignore[union-attr]
            self.running_var.fill_(1)  # type: ignore[union-attr]
            self.num_batches_tracked.zero_()  # type: ignore[union-attr,operator]

    def reset_parameters(self) -> None:
        self.reset_running_stats()
        if self.affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        super(BatchNormBase, self)._load_from_state_dict(state_dict, prefix, local_metadata,
            strict, missing_keys, unexpected_keys, error_msgs)


class BatchNorm2d(BatchNormBase):
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1,
        affine: bool = True, track_running_stats: bool = True, device=None, dtype=None):
        super(BatchNorm2d, self).__init__(num_features, eps, momentum, affine, track_running_stats, device, dtype)

    def forward(self, input: Tensor) -> Tensor:
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:  # type: ignore[has-type]
                self.num_batches_tracked.add_(1)  # type: ignore[has-type]
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        if self.training:
            bn_training = True
            output = F.batch_norm(input, self.running_mean, self.running_var, self.weight,
                            self.bias, bn_training, exponential_average_factor, self.eps)
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)
            if self.track_running_stats:
                alpha = 1.0 / input.shape[0]
                running_mean = torch.clone(self.running_mean)[None, :, None, None]
                running_var = torch.clone(self.running_var)[None, :, None, None]
                input_mean = input.mean(dim=(2,3), keepdim=True)
                input_var = input.var(dim=(2,3), keepdim=True)
                sample_mean = input_mean * alpha + (1-alpha) * running_mean
                sample_var = input_var * alpha + (1-alpha) * running_var + alpha*(1-alpha)*(input_mean-running_mean)**2
                output = (input-sample_mean)/(torch.sqrt(sample_var+self.eps)) * self.weight[None, :, None, None] + self.bias[None, :, None, None]
            else:
                output = F.batch_norm(input, None, None, self.weight,
                            self.bias, bn_training, exponential_average_factor, self.eps)
                
        return output


class ExpectedBatchNorm2d(BatchNormBase):
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1,
        affine: bool = True, device=None, dtype=None):
        track_running_stats = True
        super(ExpectedBatchNorm2d, self).__init__(num_features, eps, momentum, affine, track_running_stats, device, dtype)

    def forward(self, input: Tensor) -> Tensor:
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:  # type: ignore[has-type]
                self.num_batches_tracked.add_(1)  # type: ignore[has-type]
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        if self.training:
            # do BN
            running_mean = self.running_mean[None, :, None, None]
            running_var = self.running_var[None, :, None, None]
            output = (input-running_mean)/(torch.sqrt(running_var+self.eps)) * self.weight[None, :, None, None] + self.bias[None, :, None, None]
            # update mean and var
            input_mean = input.mean(dim=(0,2,3)).detach()
            input_var = input.var(dim=(0,2,3)).detach()
            self.running_mean = input_mean * exponential_average_factor + (1-exponential_average_factor) * self.running_mean
            self.running_var = input_var * exponential_average_factor + (1-exponential_average_factor) * self.running_var
        else:
            running_mean = torch.clone(self.running_mean)[None, :, None, None]
            running_var = torch.clone(self.running_var)[None, :, None, None]
            output = (input-running_mean)/(torch.sqrt(running_var+self.eps)) * self.weight[None, :, None, None] + self.bias[None, :, None, None]                   
        return output


class SampleExpectedBatchNorm2d(BatchNormBase):
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1,
        affine: bool = True, sample_momentum: float = 0.1, device=None, dtype=None):
        track_running_stats = True
        super(SampleExpectedBatchNorm2d, self).__init__(num_features, eps, momentum, affine, track_running_stats, device, dtype)
        self.sample_momentum = sample_momentum

    def forward(self, input: Tensor) -> Tensor:
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:  # type: ignore[has-type]
                self.num_batches_tracked.add_(1)  # type: ignore[has-type]
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # do BN
        sample_mean = input.mean(dim=(2,3), keepdim=True).detach()
        sample_var = input.var(dim=(2,3), keepdim=True).detach()
        running_mean = sample_mean * self.sample_momentum + (1-self.sample_momentum) * self.running_mean[None, :, None, None]
        running_var = sample_var * self.sample_momentum + (1-self.sample_momentum) * self.running_var[None, :, None, None]
        output = (input-running_mean)/(torch.sqrt(running_var+self.eps)) * self.weight[None, :, None, None] + self.bias[None, :, None, None]

        if self.training:
            # update mean and var
            input_mean = input.mean(dim=(0,2,3)).detach()
            input_var = input.var(dim=(0,2,3)).detach()
            self.running_mean = input_mean * exponential_average_factor + (1-exponential_average_factor) * self.running_mean
            self.running_var = input_var * exponential_average_factor + (1-exponential_average_factor) * self.running_var
    
        return output


class SampleGExpectedBatchNorm2d(BatchNormBase):
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1,
        affine: bool = True, sample_momentum: float = 0.1, device=None, dtype=None):
        track_running_stats = True
        super(SampleGExpectedBatchNorm2d, self).__init__(num_features, eps, momentum, affine, track_running_stats, device, dtype)
        self.sample_momentum = sample_momentum

    def forward(self, input: Tensor) -> Tensor:
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:  # type: ignore[has-type]
                self.num_batches_tracked.add_(1)  # type: ignore[has-type]
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # do BN
        sample_mean = input.mean(dim=(2,3), keepdim=True)
        sample_var = input.var(dim=(2,3), keepdim=True)
        running_mean = sample_mean * self.sample_momentum + (1-self.sample_momentum) * self.running_mean[None, :, None, None]
        running_var = sample_var * self.sample_momentum + (1-self.sample_momentum) * self.running_var[None, :, None, None]
        output = (input-running_mean)/(torch.sqrt(running_var+self.eps)) * self.weight[None, :, None, None] + self.bias[None, :, None, None]

        if self.training:
            # update mean and var
            input_mean = input.mean(dim=(0,2,3)).detach()
            input_var = input.var(dim=(0,2,3)).detach()
            self.running_mean = input_mean * exponential_average_factor + (1-exponential_average_factor) * self.running_mean
            self.running_var = input_var * exponential_average_factor + (1-exponential_average_factor) * self.running_var
    
        return output