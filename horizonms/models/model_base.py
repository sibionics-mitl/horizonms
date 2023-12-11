import os
import torch
from torch import nn
import torch.nn.functional as F
from abc import ABC, abstractmethod


__all__ = ("BaseModel")


class BaseModel(nn.Module, ABC):
    r"""Base class for network model.

    Args:
        net (nn.Module): Deep learning network.
        final_activation ('softmax' | 'sigmoid' | None): Decide which type of operator is used to the output of `net`.
            When final_activation=None, no operator is applied.
            When final_activation='softmax', softmax operator is applied.
            When final_activation='softmax', sigmoid operator is applied.
    """
    def __init__(self, net, final_activation=None):
        super(BaseModel, self).__init__()
        self.net = net
        assert final_activation in ["softmax", "sigmoid", None], "final_activation has to be in ['softmax', 'sigmoid', None]"
        self.final_activation = final_activation

    @abstractmethod
    def calculate_losses(self, targets, ypred):
        pass 

    @abstractmethod
    def calculate_metrics(self, targets, ypred):
        pass

    @abstractmethod
    def forward_train(self, images, targets):
        pass

    @abstractmethod
    @torch.no_grad()
    def test_one_batch(self, images, targets):
        pass

    @abstractmethod
    @torch.no_grad()
    def predict_one_batch(self, images):
        pass

    def forward(self, images):
        ypred = self.net(images)
        if isinstance(ypred, list) | isinstance(ypred, tuple):
            if self.final_activation == "softmax":
                ypred = [F.softmax(pred, dim=1) for pred in ypred]
            elif self.final_activation == "sigmoid":
                ypred = [F.sigmoid(pred) for pred in ypred]
        else:
            if self.final_activation == "softmax":
                ypred = F.softmax(ypred, dim=1)
            elif self.final_activation == "sigmoid":
                ypred = F.sigmoid(ypred)
        return ypred

    def load_model(self, model_file, data_parallel=False):
        if data_parallel: # the model is saved in data paralle mode
            self.net = torch.nn.DataParallel(self.net)

        if model_file:
            assert os.path.exists(model_file), f"{model_file} does not exist."
            # stored = torch.load(model_file, map_location=lambda storage, loc: storage)
            stored = torch.load(model_file, map_location='cpu')
            if 'state_dict' in stored.keys():
                self.net.load_state_dict(stored['state_dict'])
            else:
                self.net.load_state_dict(stored)

        if data_parallel: # convert the model back to the single GPU version
            self.net = self.net.module
