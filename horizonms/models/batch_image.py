import math
import torch
from torch import nn, Tensor
from torch.jit.annotations import List, Tuple, Dict, Optional
from .. import transforms as T


__all__ = ("BatchImage")


class BatchImage(nn.Module):
    r"""Convert a list of (input, target) into batch format such that it can be used by network.
    
    Args:
        size_divisible (int): the size of the input is converted to the ceil number which is divisible by size_divisible.
    """

    def __init__(self, size_divisible: int = 32):
        super(BatchImage, self).__init__()
        self.size_divisible = size_divisible

    def forward(self,
                images,       # type: List[Tensor]
                targets=None  # type: Optional[List[Dict[str, Tensor]]]
                ):
        # type: (...) -> Tuple[Tensor, Optional[List[Dict[str, Tensor]]]]
        images = [img for img in images]
        images = self.batch_images(images, size_divisible=self.size_divisible)

        if targets is None:
            return images, targets
        else:
            targets_batch = dict()
            for key in targets[0].keys():
                targets_batch[key] = [t[key] for t in targets]
            for key in targets_batch.keys():
                key_type = targets_batch[key][0].type
                islist = targets_batch[key][0].islist
                if islist:
                    if key_type is None:
                        value = [v.value for v in targets_batch[key]]
                    if key_type == 'labels':
                        value = [torch.stack([v.value[i] for v in targets_batch[key]]) for i in range(len(targets_batch[key][0].value))]
                    if key_type == 'masks':
                        value = [self.batch_images([v.value[i] for v in targets_batch[key]],
                                    size_divisible=self.size_divisible)
                                    for i in range(len(targets_batch[key][0].value))]
                    if key_type == 'bboxes':
                        raise ValueError("Type bboxes can not be islist=True in target")
                else:
                    if key_type is None:
                        value = [v.value for v in targets_batch[key]]
                    if key_type == 'labels':
                        value = torch.stack([v.value for v in targets_batch[key]])
                    if key_type == 'masks':
                        value = self.batch_images([v.value for v in targets_batch[key]],
                                    size_divisible=self.size_divisible)
                    if key_type == 'bboxes':
                        value = [v.value for v in targets_batch[key]]
                targets_batch[key] = T.TargetStructure(type=key_type, value=value)
            return images, targets_batch

    def max_by_axis(self, the_list):
        # type: (List[List[int]]) -> List[int]
        maxes = the_list[0]
        for sublist in the_list[1:]:
            for index, item in enumerate(sublist):
                maxes[index] = max(maxes[index], item)
        return maxes

    def batch_images(self, images, size_divisible=32):
        max_size = self.max_by_axis([list(img.shape) for img in images])
        stride = float(size_divisible)
        max_size = list(max_size)
        max_size[1] = int(math.ceil(float(max_size[1]) / stride) * stride)
        max_size[2] = int(math.ceil(float(max_size[2]) / stride) * stride)

        batch_shape = [len(images)] + max_size
        batched_imgs = images[0].new_full(batch_shape, 0)
        for img, pad_img in zip(images, batched_imgs):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)

        return batched_imgs
