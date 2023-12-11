import numpy as np
import pandas as pd
from utils import *

from horizonms.transforms import Normalizer, Hue, Saturation, Brightness, GaussianBlur, GaussianNoise, \
    Invert, Solarize, Equalize, Lighting, Contrast, AutoContrast, Posterize, Sharpness
from horizonms.transforms import CVNormalizer, CVHue, CVSaturation, CVBrightness, CVGaussianBlur, \
    CVGaussianNoise, CVInvert, CVSolarize, CVEqualize, CVLighting, CVContrast, CVAutoContrast, CVPosterize, CVSharpness

from horizonms.transforms import RandomHue, RandomSaturation, RandomBrightness, RandomGaussianBlur, \
    RandomGaussianNoise, RandomInvert, RandomSolarize, RandomEqualize, RandomLighting, RandomContrast, \
    RandomAutoContrast, RandomPosterize, RandomSharpness
from horizonms.transforms import CVRandomHue, CVRandomSaturation, CVRandomBrightness, \
    CVRandomGaussianBlur, CVRandomGaussianNoise, CVRandomInvert, CVRandomSolarize, CVRandomEqualize, CVRandomLighting, \
    CVRandomContrast, CVRandomAutoContrast, CVRandomPosterize, CVRandomSharpness

if __name__ == "__main__":
    image = {
        # 'Hue': {'torch': Hue, 'cv': CVHue, 'params': {'hue_factor': urandom_choice(
        #     [urandom_uniform(low=-0.5, high=0.5), urandom_uniform_n(low=-0.5, high=0.5, n=10), (-0.5, 0.5)])},
        #         'gray_supported': False, 'target_type': ['bboxes', 'points', 'masks', 'labels', None]},  # 灰度图不可用
        # 'Saturation': {'torch': Saturation, 'cv': CVSaturation, 'params': {'saturation_factor': urandom_choice(
        #     [urandom_uniform(low=0, high=1), urandom_uniform_n(low=0, high=1, n=10), (0, 1)])},
        #                'gray_supported': False, 'target_type': ['bboxes', 'points', 'masks', 'labels', None]},  # 灰度图不可用
        # 'Brightness': {'torch': Brightness, 'cv': CVBrightness, 'params': {'brightness_factor': urandom_choice(
        #     [urandom_uniform(low=0, high=2), urandom_uniform_n(low=0, high=2, n=10), (0, 2)])},
        #                'gray_supported': True, 'target_type': ['bboxes', 'points', 'masks', 'labels', None]},
        # 'GaussianBlur': {'torch': GaussianBlur, 'cv': CVGaussianBlur, 'params': {'sigma': 30.0, 'kernel_size': 7},
        #                  'gray_supported': True, 'target_type': ['bboxes', 'points', 'masks', 'labels', None]},
        # 'GaussianNoise': {'torch': GaussianNoise, 'cv': CVGaussianNoise, 'params': {
        #     'std': urandom_choice([urandom_uniform(low=0, high=1), urandom_uniform_n(low=0, high=1, n=10), (0, 1)]),
        #     'mean': 0.0},
        #                   'gray_supported': True, 'target_type': ['bboxes', 'points', 'masks', 'labels', None]},
        # 'Solarize': {'torch': Solarize, 'cv': CVSolarize, 'params': {'solarize_threshold': urandom_choice(
        #     [urandom_uniform(low=0, high=1), urandom_uniform_n(low=0, high=1, n=10), (0, 1)])},
        #              'gray_supported': True, 'target_type': ['bboxes', 'points', 'masks', 'labels', None]},
        # 'Contrast': {'torch': Contrast, 'cv': CVContrast, 'params': {'contrast_factor': urandom_choice(
        #     [urandom_uniform(low=0, high=2), urandom_uniform_n(low=0, high=2, n=10), (0, 2)])}, 'gray_supported': True,
        #              'target_type': ['bboxes', 'points', 'masks', 'labels', None]},
        # 'Posterize': {'torch': Posterize, 'cv': CVPosterize, 'params': {'posterize_bins': 1}, 'gray_supported': True,
        #               'target_type': ['bboxes', 'points', 'masks', 'labels', None]},
        'Sharpness': {'torch': Sharpness, 'cv': CVSharpness, 'params': {'sharpness_factor': urandom_choice(
            [urandom_uniform(low=0, high=10), urandom_uniform_n(low=0, high=10, n=10), (0, 10)])},
                      'gray_supported': True, 'target_type': ['bboxes', 'points', 'masks', 'labels', None]},
        # 'Equalize': {'torch': Equalize, 'cv': CVEqualize, 'params': {}, 'gray_supported': True,
        #              'target_type': ['bboxes', 'points', 'masks', 'labels', None]},
        # 'Invert': {'torch': Invert, 'cv': CVInvert, 'params': {}, 'gray_supported': True,
        #            'target_type': ['bboxes', 'points', 'masks', 'labels', None]},
        # 'AutoContrast': {'torch': AutoContrast, 'cv': CVAutoContrast, 'params': {}, 'gray_supported': True,
        #                  'target_type': ['bboxes', 'points', 'masks', 'labels', None]},
        # 'Normalizer_zscore': {'torch': Normalizer, 'cv': CVNormalizer,
        #                       'params': {"mode": 'zscore', 'image_base': True}, 'gray_supported': True,
        #                       'target_type': ['bboxes', 'points', 'masks', 'labels', None]},
        # 'Normalizer_zero-one': {'torch': Normalizer, 'cv': CVNormalizer,
        #                         'params': {"mode": 'zero-one', 'image_base': True}, 'gray_supported': True,
        #                         'target_type': ['bboxes', 'points', 'masks', 'labels', None]},
        # 'Normalizer_negative-positive-one': {'torch': Normalizer, 'cv': CVNormalizer,
        #                                      'params': {"mode": 'negative-positive-one', 'image_base': True},
        #                                      'gray_supported': True,
        #                                      'target_type': ['bboxes', 'points', 'masks', 'labels', None]},
        # 'Lighting': {'torch': Lighting, 'cv': CVLighting, 'params': {'alphastd': urandom_choice(
        #     [urandom_uniform(low=0, high=1), urandom_uniform_n(low=0, high=1, n=10), (0, 1)]),
        #     'eigval': np.array([0.5, 0.0188, 0.0045]),
        #     'eigvec': np.array([
        #         [-0.5675, 0.7192, 0.4009],
        #         [-0.5808, -0.0045, -0.8140],
        #         [-0.5836, -0.6948, 0.4203],
        #     ])}, 'gray_supported': False, 'target_type': ['bboxes', 'points', 'masks', 'labels', None]},
    }

    random_image = {
        # 'Hue': {'torch': RandomHue, 'cv': CVRandomHue, 'params': {'prob': 1, "hue_factor": urandom_choice(
        #     [urandom_uniform(low=-0.5, high=0.5), urandom_uniform_n(low=-0.5, high=0.5, n=10), (-0.5, 0.5)])},
        #         'gray_supported': False, 'target_type': ['bboxes', 'points', 'masks', 'labels', None]},
        # 'Saturation': {'torch': RandomSaturation, 'cv': CVRandomSaturation,
        #                'params': {'prob': 1, "saturation_factor": urandom_choice(
        #                    [urandom_uniform(low=0, high=1), urandom_uniform_n(low=0, high=1, n=10), (0, 1)])},
        #                'gray_supported': False, 'target_type': ['bboxes', 'points', 'masks', 'labels', None]},
        # 'Brightness': {'torch': RandomBrightness, 'cv': CVRandomBrightness,
        #                'params': {'prob': 1, "brightness_factor": urandom_choice(
        #                    [urandom_uniform(low=0, high=1), urandom_uniform_n(low=0, high=1, n=10), (0, 1)])},
        #                'gray_supported': True, 'target_type': ['bboxes', 'points', 'masks', 'labels', None]},
        # 'GaussianBlur': {'torch': RandomGaussianBlur, 'cv': CVRandomGaussianBlur,
        #                  'params': {'prob': 1, "sigma": urandom_choice(
        #                      [urandom_uniform(low=0, high=100), urandom_uniform_n(low=0, high=100, n=10), (0, 100)]),
        #                             'kernel_size': (4, 21)}, 'gray_supported': True,
        #                  'target_type': ['bboxes', 'points', 'masks', 'labels', None]},
        # 'GaussianNoise': {'torch': RandomGaussianNoise, 'cv': CVRandomGaussianNoise,
        #                   'params': {'prob': 1, "std": urandom_choice(
        #                       [urandom_uniform(low=0, high=1), urandom_uniform_n(low=0, high=1, n=10), (0, 1)]),
        #                              'mean': 0.0}, 'gray_supported': True,
        #                   'target_type': ['bboxes', 'points', 'masks', 'labels', None]},
        # 'Solarize': {'torch': RandomSolarize, 'cv': CVRandomSolarize,
        #              'params': {'prob': 1, "solarize_threshold": urandom_choice(
        #                  [urandom_uniform(low=0, high=1), urandom_uniform_n(low=0, high=1, n=10), (0, 1)])},
        #              'gray_supported': True, 'target_type': ['bboxes', 'points', 'masks', 'labels', None]},
        # 'Contrast': {'torch': RandomContrast, 'cv': CVRandomContrast,
        #              'params': {'prob': 1, "contrast_factor": urandom_choice(
        #                  [urandom_uniform(low=0, high=2), urandom_uniform_n(low=0, high=2, n=10), (0, 2)])},
        #              'gray_supported': True, 'target_type': ['bboxes', 'points', 'masks', 'labels', None]},
        # 'Posterize': {'torch': RandomPosterize, 'cv': CVRandomPosterize,
        #               'params': {'prob': 1, "posterize_bins": (1, 8)}, 'gray_supported': True,
        #               'target_type': ['bboxes', 'points', 'masks', 'labels', None]},
        # 'Sharpness': {'torch': RandomSharpness, 'cv': CVRandomSharpness,
        #               'params': {'prob': 1, "sharpness_factor": urandom_choice(
        #                   [urandom_uniform(low=0, high=10), urandom_uniform_n(low=0, high=10, n=10), (0, 10)])},
        #               'gray_supported': True, 'target_type': ['bboxes', 'points', 'masks', 'labels', None]},
        # 'Equalize': {'torch': RandomEqualize, 'cv': CVRandomEqualize, 'params': {'prob': 1},
        #              'gray_supported': True, 'target_type': ['bboxes', 'points', 'masks', 'labels', None]},
        # 'Invert': {'torch': RandomInvert, 'cv': CVRandomInvert, 'params': {'prob': 1}, 'gray_supported': True,
        #            'target_type': ['bboxes', 'points', 'masks', 'labels', None]},
        # 'AutoContrast': {'torch': RandomAutoContrast, 'cv': CVRandomAutoContrast, 'params': {'prob': 1},
        #                  'gray_supported': True, 'target_type': ['bboxes', 'points', 'masks', 'labels', None]},
        # 'Lighting': {'torch': RandomLighting, 'cv': CVRandomLighting, 'params': {'prob': 1},
        #              'gray_supported': False, 'target_type': ['bboxes', 'points', 'masks', 'labels', None]}
    }

    img_pd = pd.read_csv('../../data/test_color_img.csv')
    img_path = '../../data/test_color_img/'
    save_dir = '../../results/testing_image/'
    data_type_list = ['float', 'uint8']
    # data_type_list = ['uint8']
    test_dict = {'image': image, 'random_image': random_image}
    # test_dict = {'random_image': random_image}

    for data_type in data_type_list:
        print(data_type)
        for key in test_dict.keys():
            msg_pd = error_calculation(test_dict[key], img_pd, img_path, os.path.join(save_dir, data_type, key),
                                       data_type)
            msg_pd.to_csv(os.path.join(save_dir, '{}_{}.csv'.format(key, data_type)), index=False, encoding='utf-8-sig')
