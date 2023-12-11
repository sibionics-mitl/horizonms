import torch
from abc import ABC, abstractmethod
from ..model_base import BaseModel
from ...builder import MODELS


__all__ = ("BaseDetection")


@MODELS.register_module()
class BaseDetection(BaseModel, ABC):
    r"""Base class for object detection task.

    Args:
        net (nn.Module): Deep learning network.
        batch_image: class used to convert a list of (input, target) into batch format used in network training and testing.
        divisible: it determines the size of the batched input such that it is divisible by `divisible` and larger than the size of the input.
        batch_transforms: batch transformation for network training.
    """
    def __init__(self, net, batch_image, divisible=1, batch_transforms=None):
        super(BaseDetection, self).__init__(net, None)
        self.batch_image = None
        if batch_image is not None:
            self.batch_image = batch_image(divisible)

        self.batch_transforms = batch_transforms

    def preprocessing_input(self, images, targets=None):
        if self.batch_image is not None:
            images, targets = self.batch_image(images, targets)
        if self.batch_transforms is not None:
            images, targets = self.batch_transforms(images, targets)
        return images, targets

    @abstractmethod
    def calculate_losses(self, targets, ypred):
        pass 

    @abstractmethod
    def calculate_metrics(self, targets, ypred):
        pass

    def forward_train(self, images, targets):
        images, targets = self.preprocessing_input(images, targets)
        if torch.isnan(images).sum()>0:
            print('image is nan ..............')
        if torch.isinf(images).sum()>0:
            print('image is inf ..............')   
        ypred = self.forward(images)

        losses = dict()
        if targets is None:
            return losses, ypred
            
        losses = self.calculate_losses(targets, ypred)
        return losses, ypred

    @torch.no_grad()
    def test_one_batch(self, images, targets):
        pass

    @abstractmethod
    def predict_one_batch(self, images):
        pass