import numpy as np
import random
from torch.utils.data import Sampler
from torch.utils.data import Dataset
from typing import Iterator, List, Union


__all__ = ("BalancedSampler")


class BalancedSampler(Sampler[int]):
    r"""Sampling elements such that the number of samples in different classes is balanced based on a predefined sampling rule. 
    
    Args:
        dataset (Dataset): dataset to sample from.
        samples_per_class (Union[List[int],str]): sampling rule. It is 'square_root' (default), 'equal' or list of int, 
            which defines the number of samples in each class.
        sampling_classes (int): number of classes in the dataset.
    """
    def __init__(self, dataset: Dataset, samples_per_class: Union[List[int],str] = 'square_root',
                 sampling_classes: int = None) -> None:
        self.dataset = dataset
        self.num_samples = len(self.dataset)
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


    def __iter__(self) -> Iterator[int]:
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
        yield from iter(indices)

    def __len__(self) -> int:
        return self.num_samples