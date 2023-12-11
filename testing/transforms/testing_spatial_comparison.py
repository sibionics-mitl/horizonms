import pandas as pd
from utils import *

from horizonms.transforms import Fliplr, CropX, CropY, Flipud, Rotate, TranslateX, TranslateY, ShearX, \
    ShearY, RandomResizedCrop, ImagePadding, ResizeWidth, Scale
from horizonms.transforms import CVFliplr, CVCropX, CVCropY, CVFlipud, CVRotate, CVTranslateX, \
    CVTranslateY, CVShearX, CVShearY, CVRandomResizedCrop, CVImagePadding, CVResizeWidth, CVScale

from horizonms.transforms import RandomCropX, RandomCropY, RandomRotate, RandomTranslateX, \
    RandomTranslateY, RandomShearX, RandomShearY,Resize, RandomScale
from horizonms.transforms import CVRandomCropX, CVRandomCropY, CVRandomRotate, CVRandomTranslateX, \
    CVRandomTranslateY, CVRandomShearX, CVRandomShearY,CVResize, CVRandomScale


if __name__ == "__main__":

    spatial = {
    #     'Fliplr': {'torch': Fliplr, 'cv': CVFliplr, 'params': {}, 'gray_supported': True,'target_type':['bboxes', 'points', 'masks', 'labels', None]},
    #     'Flipud': {'torch': Flipud, 'cv': CVFlipud, 'params': {}, 'gray_supported': True,'target_type':['bboxes', 'points', 'masks', 'labels', None]},
    #     'TranslateX': {'torch': TranslateX, 'cv': CVTranslateX,
    #                    'params': {'translate_ratio': urandom_choice([urandom_uniform(low=0, high=1 ), urandom_uniform_n(low=0, high=1, n=10 ), (0, 1)]),
    #                               'fill': 200},
    #                    'gray_supported': True,'target_type':['bboxes', 'points', 'masks', 'labels', None]},
    #     'TranslateY': {'torch': TranslateY, 'cv': CVTranslateY,
    #                    'params': {'translate_ratio': urandom_choice([urandom_uniform(low=0, high=1 ), urandom_uniform_n(low=0, high=1, n=10 ), (0, 1)]),
    #                               'fill': 50},
    #                    'gray_supported': True,'target_type':['bboxes', 'points', 'masks', 'labels', None]},
    #     'Rotate': {'torch': Rotate, 'cv': CVRotate,
    #                'params': {'rotate_degree': urandom_choice([urandom_uniform(low=0, high=360 ), urandom_uniform_n(low=0, high=360, n=10 ), (0, 360)]),
    #                           'fill': 150},
    #                'gray_supported': True, 'target_type':['masks', 'labels', None]},
    #     'ShearX': {'torch': ShearX, 'cv': CVShearX,
    #                'params': {'shear_degree': urandom_choice([urandom_uniform(low=-180, high=180 ), urandom_uniform_n(low=-180, high=180, n=10 ), (-180, 180)]),
    #                           'fill': 50},
    #                'gray_supported': True,'target_type':['masks', 'labels', None]},
    #     'ShearY': {'torch': ShearY, 'cv': CVShearY,
    #                'params': {'shear_degree': urandom_choice([urandom_uniform(low=-180, high=180 ), urandom_uniform_n(low=-180, high=180, n=10 ), (-180, 180)]),
    #                           'fill': 250},
    #                'gray_supported': True,'target_type':['masks', 'labels', None]},
        'RandomResizedCrop': {'torch': RandomResizedCrop, 'cv': CVRandomResizedCrop, 'params': {'size': 300},
                              'gray_supported': True,'target_type':[ 'masks']},
    #     'ImagePadding': {'torch': ImagePadding, 'cv': CVImagePadding, 'params': {}, 'gray_supported': True,'target_type':['bboxes', 'points', 'masks', 'labels', None]},
    #     'ResizeWidth': {'torch': ResizeWidth, 'cv': CVResizeWidth, 'params': {'width': urandom_uniform(low=100, high=300), 'min_size_list': [300]},
    #                     'gray_supported': True,'target_type':['bboxes', 'points', 'masks', 'labels', None]},
    #     'Resize': {'torch': Resize, 'cv': CVResize,
    #                'params': { 'size': (urandom_int(low=100, high=300),urandom_int(low=100, high=300))},
    #                'gray_supported': True,'target_type':['bboxes', 'points', 'masks', 'labels', None]},
    #     'CropX': {'torch': CropX, 'cv': CVCropX,
    #               'params': {'crop_ratio': urandom_choice(
    #                   [urandom_uniform(low=0, high=1), urandom_uniform_n(low=0, high=1, n=10), (0, 1)])},
    #               'gray_supported': True,'target_type':['bboxes', 'points', 'masks', 'labels', None]},
    #     'CropY': {'torch': CropY, 'cv': CVCropY,
    #               'params': {'crop_ratio': urandom_choice(
    #                   [urandom_uniform(low=0, high=1), urandom_uniform_n(low=0, high=1, n=10), (0, 1)])},
    #               'gray_supported': True,'target_type':['bboxes', 'points', 'masks', 'labels', None]},
    #     'Scale': {'torch': Scale, 'cv': CVScale,
    #               'params': {'scale_range': urandom_choice([urandom_uniform(low=0, high=1), urandom_uniform_n(low=0, high=1, n=10), (0, 1)]),
    #                          'scale_width': urandom_choice([True, False]),
    #                          'scale_height': urandom_choice([True, False]),
    #                          'scale_same': urandom_choice([True, False])},
    #               'gray_supported': True, 'target_type': ['masks', 'labels', None, 'bboxes', 'points']}
    }
    random_spatial = {
        # 'TranslateX': {'torch': RandomTranslateX, 'cv': CVRandomTranslateX,
        #                'params': {'prob': 1,
        #                           'translate_ratio': urandom_choice(
        #                               [urandom_uniform(low=0, high=1), urandom_uniform_n(low=0, high=1, n=10),
        #                                (0, 1)])},
        #                'gray_supported': True,'target_type':['bboxes', 'points', 'masks', 'labels', None]},
        # 'TranslateY': {'torch': RandomTranslateY, 'cv': CVRandomTranslateY,
        #                'params': {'prob': 1,
        #                           'translate_ratio': urandom_choice(
        #                               [urandom_uniform(low=0, high=1), urandom_uniform_n(low=0, high=1, n=10),
        #                                (0, 1)])},
        #                'gray_supported': True,'target_type':['bboxes', 'points', 'masks', 'labels', None]},
        # 'Rotate': {'torch': RandomRotate, 'cv': CVRandomRotate,
        #            'params': {'prob': 1,
        #                       'rotate_degree': urandom_choice(
        #                           [urandom_uniform(low=0, high=360), urandom_uniform_n(low=0, high=360, n=10),
        #                            (0, 360)]),
        #                       'fill': 100},
        #            'gray_supported': True,'target_type':['masks', 'labels', None]},
        # 'ShearX': {'torch': RandomShearX, 'cv': CVRandomShearX,
        #            'params': {'prob': 1,
        #                       'shear_degree': urandom_choice(
        #                           [urandom_uniform(low=-180, high=180), urandom_uniform_n(low=-180, high=180, n=10),
        #                            (-180, 180)]),
        #                       'fill': 100},
        #            'gray_supported': True,'target_type':['masks', 'labels', None]},
        # 'ShearY': {'torch': RandomShearY, 'cv': CVRandomShearY,
        #            'params': {'prob': 1,
        #                       'shear_degree': urandom_choice(
        #                           [urandom_uniform(low=-180, high=180), urandom_uniform_n(low=-180, high=180, n=10),
        #                            (-180, 180)]),
        #                       'fill': 100},
        #            'gray_supported': True,'target_type':['masks', 'labels', None]},
        # 'CropX': {'torch': RandomCropX, 'cv': CVRandomCropX,
        #           'params': {'prob': 1, 'crop_ratio': urandom_choice(
        #               [urandom_uniform(low=0, high=1), urandom_uniform_n(low=0, high=1, n=10), (0, 1)])},
        #           'gray_supported': True,'target_type':['bboxes', 'points', 'masks', 'labels', None]},
        # 'CropY': {'torch': RandomCropY, 'cv': CVRandomCropY,
        #           'params': {'prob': 1, 'crop_ratio': urandom_choice(
        #               [urandom_uniform(low=0, high=1), urandom_uniform_n(low=0, high=1, n=10), (0, 1)])},
        #           'gray_supported': True,'target_type':['bboxes', 'points', 'masks', 'labels', None]},
        # 'Scale': {'torch': RandomScale, 'cv': CVRandomScale,
        #           'params': {'scale_range': urandom_choice([urandom_uniform(low=0, high=1), urandom_uniform_n(low=0, high=1, n=10), (0, 1)]),
        #                      'scale_width': urandom_choice([True, False]),
        #                      'scale_height': urandom_choice([True, False]),
        #                      'scale_same': urandom_choice([True, False]),
        #                      'prob': 1.0},
        #           'gray_supported': True, 'target_type': ['masks', 'labels', None, 'bboxes', 'points']}
    }
    img_pd = pd.read_csv('../../data/test_color_img.csv')
    img_path = '../../data/test_color_img/'
    save_dir = '../../results/testing_spatial/'
    data_type_list = ['float']
    # data_type_list = ['uint8']
    test_dict = {'spatial': spatial, 'random_spatial': random_spatial}
    for data_type in data_type_list:
        print(data_type)
        for key in test_dict.keys():
            msg_pd = error_calculation(test_dict[key], img_pd, img_path, os.path.join(save_dir, data_type, key),
                                       data_type)
            msg_pd.to_csv(os.path.join(save_dir, '{}_{}.csv'.format(key, data_type)), index=False, encoding='utf-8-sig')
