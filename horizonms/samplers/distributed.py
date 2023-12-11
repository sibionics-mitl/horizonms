import math
import numpy as np
import random
import torch
from torch.utils.data import Dataset
from typing import List, Union


__all__ = ("DistributedWeightedSampler", "DistributedBalancedSampler")


class DistributedWeightedSampler(torch.utils.data.distributed.DistributedSampler):
    r"""Sampler that restricts data loading to a subset of the dataset.
    It samples elements such that the number of samples in different classes is determined by weights. 
    
    Args:
        dataset (Dataset): dataset to sample from.
        weights (Union[List[float],str]): It is 'square_root' (default), 'equal' or list of int, 
            which defines the number of samples in each class.
        sampling_classes (int): number of classes in the dataset.
    """
    def __init__(self, dataset: Dataset, weights: Union[List[float],str] = 'square_root_inverse',
                 replacement: bool = True, sampling_classes: int = None, *argv, **kwargs) -> None:
        super(DistributedWeightedSampler, self).__init__(dataset=dataset, *argv, **kwargs)
        if sampling_classes is None:
            sampling_classes = self.dataset.sampling_classes
        if weights == 'square_root_inverse':
            sampling_labels = np.array(self.dataset.sampling_labels)
            nb_samples = np.asarray([np.sum(sampling_labels==k) for k in range(sampling_classes)])
            weights = 1 / np.sqrt(nb_samples)
            weights = weights / weights.sum()
            # print(f"squre root weights = {weights}")
            weights = weights[sampling_labels]
            self.weights = torch.tensor(weights)
        elif weights == 'inverse':
            sampling_labels = np.array(self.dataset.sampling_labels)
            nb_samples = np.asarray([np.sum(sampling_labels==k) for k in range(sampling_classes)])
            weights = 1.0 / nb_samples
            weights = weights / weights.sum()
            # print(f"inverse weights = {weights}")
            weights = weights[sampling_labels]
            self.weights = torch.tensor(weights)
        elif isinstance(weights, list):
            self.weights = torch.tensor(weights)    
        else:
            raise ValueError("weights has to be 'square_root_inverse' (default), 'inverse' or list of float")
        assert len(self.weights) == len(self.dataset), "weights and len(dataset) are not equal"
        
        self.replacement = replacement

    def __iter__(self):
        indices = torch.multinomial(self.weights, len(self.dataset), self.replacement).tolist()

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size, f"len = {len(indices)}, size = {self.total_size}"

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)


class DistributedBalancedSampler(torch.utils.data.distributed.DistributedSampler):
    r"""Sampler that restricts data loading to a subset of the dataset.
    It samples elements such that the number of samples in different classes is balanced based on a predefined sampling rule. 
    
    Args:
        dataset (Dataset): dataset to sample from.
        samples_per_class (Union[List[int],str]): sampling rule. It is 'square_root' (default), 'equal' or list of int, 
            which defines the number of samples in each class.
        sampling_classes (int): number of classes in the dataset.
    """
    def __init__(self, dataset: Dataset, samples_per_class: Union[List[int],str] = 'square_root',
                 sampling_classes: int = None, *argv, **kwargs) -> None:
        super(DistributedBalancedSampler, self).__init__(dataset=dataset, *argv, **kwargs)
        if sampling_classes is None:
            sampling_classes = self.dataset.sampling_classes
        if samples_per_class == 'square_root':
            sampling_labels = np.array(self.dataset.sampling_labels)
            nb_samples = np.asarray([np.sum(sampling_labels==k) for k in range(sampling_classes)])
            nb_samples_per_class = np.sqrt(nb_samples)
            self.nb_samples_per_class = np.around(nb_samples_per_class / nb_samples_per_class.sum() \
                                        * nb_samples.sum()).astype(np.int32)
        elif samples_per_class == 'equal':
            sampling_labels = np.array(self.dataset.sampling_labels)
            nb_samples = np.asarray([np.sum(sampling_labels==k) for k in range(sampling_classes)])
            nb_samples_per_class = np.array([1] * sampling_classes)
            self.nb_samples_per_class = np.around(nb_samples_per_class / nb_samples_per_class.sum() \
                                        * nb_samples.sum()).astype(np.int32)
        elif isinstance(samples_per_class, list):
            self.nb_samples_per_class = samples_per_class    
        else:
            raise ValueError("samples_per_class has to be 'square_root' (default), 'equal' or list of int")
        assert len(self.nb_samples_per_class) == sampling_classes, "len(nb_samples_per_class) should be equal to the number of sampling classes"

    def __iter__(self):
        indices = []
        for c, nb_sample in enumerate(self.nb_samples_per_class):
            indices_c = np.where(np.array(self.dataset.sampling_labels)==c)[0]
            nb_c = len(indices_c)
            if nb_c >= nb_sample:
                indices += random.sample(indices_c.tolist(), k=nb_sample)
            else:
                r = nb_sample // nb_c
                indices += list(indices_c) * r
                indices += random.sample(indices_c.tolist(), k=nb_sample-nb_c*r)
        indices = random.sample(indices, k=len(indices))

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices) and padding_size > 0:
                indices += indices[:padding_size]
            elif padding_size < 0:
                indices = indices[:self.total_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size, f"len = {len(indices)}, size = {self.total_size}"

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)