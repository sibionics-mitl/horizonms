from typing import List
from torch import Tensor
import numbers
import numpy as np
from collections.abc import Sequence
import random


def _input_check_value_range_set(input, dtype='float'):
    assert isinstance(input, numbers.Number) | isinstance(input, list) | isinstance(input, tuple)
    if isinstance(input, tuple):
        assert len(input) == 2, "Both lowest and highest values have to be provided for range."
    return input


def _input_get_value_range_set(input, dtype='float'):
    if isinstance(input, tuple):
        if dtype == 'int':
            assert input[0] < input[1], "(lowest value, highest value) should be provided for range, but got (highest value, lowest value)."
            input = random.randint(input[0], input[1])
        else:
            input = random.uniform(input[0], input[1])
    elif isinstance(input, list):
        input = random.choice(input)

    if dtype == 'int':
        input = int(input)
    return input


def _gaussian_kernel_size_check_value_range_set(kernel_size):
    assert isinstance(kernel_size, int) | isinstance(kernel_size, list) | isinstance(kernel_size, tuple)
    if isinstance(kernel_size, int):
        if kernel_size < 3:
            raise ValueError("Kernel size value must be at least 3.")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("Kernel size value should be an odd and positive number.")
    elif isinstance(kernel_size, tuple):
        assert len(kernel_size) == 2, "Both minimum and maximum have to be provided for range."
        assert kernel_size>=(3,3), "The mininum value should be at least 3."
    elif isinstance(kernel_size, list):  ## 注意:对应使用列表和tuple来区别可能需要在readme说清楚,不然使用者可能会混用两者
        assert all([k>=3 for k in kernel_size]), "Kernel size must be at least 3."
        assert all([k>0 and k%2!=0 for k in kernel_size]), "Kernel size value should be an odd and positive number."
       

def _gaussian_kernel_size_get_value_range_set(kernel_size):
    if isinstance(kernel_size, tuple):
        if kernel_size[0] % 2 == 0:
            kernel_size = random.randrange(kernel_size[0]+1, kernel_size[1]+1, 2)
        else:
            kernel_size = random.randrange(kernel_size[0], kernel_size[1]+1, 2)
    elif isinstance(kernel_size, list):  ## 注意:对应使用列表和tuple来区别可能需要在readme说清楚,不然使用者可能会混用两者
        kernel_size = random.choice(kernel_size)
    kernel_size = int(kernel_size)
    return kernel_size


def setup_size(size, error_msg):
    if isinstance(size, numbers.Number):
        return int(size), int(size)

    if isinstance(size, Sequence) and len(size) == 1:
        return size[0], size[0]

    if len(size) != 2:
        raise ValueError(error_msg)

    return size


def get_image_size(img: Tensor) -> List[int]:
    # Returns (w, h) of tensor image
    return [img.shape[-1], img.shape[-2]]


def cv_image_shift(image, shift_xy):
    shift_x, shift_y = shift_xy
    height, width = image.shape[:2]
    shifted_image = np.zeros(image.shape, dtype=image.dtype)
    if shift_x>=0 and shift_y>=0:
        shifted_image[int(shift_y):,int(shift_x):,:] = image[:height-int(shift_y),:width-int(shift_x),:]
    elif shift_x>=0 and shift_y<0:
        shifted_image[:height+int(shift_y),int(shift_x):,:] = image[-int(shift_y):,:width-int(shift_x),:]
    elif shift_x <0 and shift_y >=0:
        shifted_image[int(shift_y):,:width+int(shift_x),:] = image[:height-int(shift_y),-int(shift_x):,:]
    elif shift_x<0 and shift_y<0:
        shifted_image[:height+int(shift_y),:width+int(shift_x),:] = image[-int(shift_y):,-int(shift_x):,:]
    return shifted_image
