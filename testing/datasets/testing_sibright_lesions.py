import os, sys
import numpy as np

sys.path.insert(0, os.getcwd())
import torch
from configs.config_sibright_lesions import get_experiment_config
from horizonms import transforms as T
from aimachine.datasets_utils.sibright_lesions import get_aug_sibright_lesions

from horizonms.datasets import SibrightLesionsClassificationPng

if __name__ == "__main__":
    net_name = 'efficientnet_multiheads_b3'
    scheduler = 'OneCycleLR'
    exp_config = get_experiment_config(net_name, scheduler)
    data_params = exp_config['data_params']
    dataset_params = exp_config['dataset']
    data_params['trivalaugment_pt'] = dict(
        augment_operators=[dict(type='Identity'),
                           # dict(type='Contrast', params=dict(param_range=[0.5, 2.0])),
                           # dict(type='Posterize', params=dict(param_range=[2, 8])),  #
                           dict(type='Solarize', params=dict(param_range=[1.0, 0.85])),  #
                           dict(type='Brightness', params=dict(param_range=[1.0, 0.85])),  #
                           # dict(type='AutoContrast'),
                           dict(type='Equalize'),  #
                           # dict(type='TranslateX', params=dict(param_range=[-0.05, 0.05])),  #
                           # dict(type='TranslateY', params=dict(param_range=[-0.025, 0.025])),  #
                           dict(type='CropX', params=dict(param_range=[-0.1, 0.1])),
                           dict(type='CropY', params=dict(param_range=[-0.05, 0.05])),
                           dict(type='Fliplr'), dict(type='Flipud'),
                           ],
        num_magnitude_bins=32)
    data_params['trivalaugment_cv'] = dict(
        augment_operators=[dict(type='CVIdentity'),
                           # dict(type='CVContrast', params=dict(param_range=[0.5, 2.0])),
                           # dict(type='CVPosterize', params=dict(param_range=[2, 8])),  #
                           dict(type='CVSolarize', params=dict(param_range=[1.0, 0.85])),  #
                           dict(type='CVBrightness', params=dict(param_range=[1.0, 0.85])),  #
                           # dict(type='CVAutoContrast'),
                           dict(type='CVEqualize'),  #
                           # dict(type='CVTranslateX', params=dict(param_range=[-0.05, 0.05])),  #
                           # dict(type='CVTranslateY', params=dict(param_range=[-0.025, 0.025])),  #
                           dict(type='CVCropX', params=dict(param_range=[-0.1, 0.1])),
                           dict(type='CVCropY', params=dict(param_range=[-0.05, 0.05])),
                           dict(type='CVFliplr'), dict(type='CVFlipud'),
                           ],
        num_magnitude_bins=32)


    def get_transform(train: bool, type='trivalaugment_pt'):
        transforms = []
        if train:
            trivalaugment = data_params[type]
            augment_operators = []
            for augment in trivalaugment['augment_operators']:
                if augment.get('params', False):
                    augment_operators.append(
                        T.OpStructure(type=augment['type'], value=T.OpParamStructure(**augment['params'])))
                else:
                    augment_operators.append(T.OpStructure(type=augment['type']))
            transforms.append(T.CustomizedTrivialAugment(augment_operators, trivalaugment['num_magnitude_bins']))
        # if type== 'trivalaugment_pt':
        #     transforms.append(T.Normalizer(mode='zero-one'))##'zero-one'的Normalizer没有差异
        # elif type== 'trivalaugment_cv':
        #     transforms.append(T.CVNormalizer(mode='zero-one'))
        return transforms


    preprocessing_transforms = None
    if data_params.get('preprocessing', False):
        pre_operators = []
        for op_setting in data_params['preprocessing']:
            if op_setting.get('params', False):
                pre_operators.append(T.OpStructure(type=op_setting['type'], value=op_setting['params']))
            else:
                pre_operators.append(T.OpStructure(type=op_setting['type']))
        preprocessing_transforms = T.GetPreprocessing(pre_operators)
    print(f"preprocessing_transforms = {preprocessing_transforms}")

    # Data loading code
    print("Loading data")
    mode = "train"
    if mode == 'train':
        csv_file = dataset_params['train_csv']
        transforms_pt = get_transform(True, 'trivalaugment_pt')
        transforms_cv = get_transform(True, 'trivalaugment_cv')
    else:
        csv_file = dataset_params['valid_csv']
        transforms_pt = get_transform(False, 'trivalaugment_pt')
        transforms_cv = get_transform(False, 'trivalaugment_cv')
    print(transforms_pt)
    print(transforms_cv)
    print(dataset_params['root_path'], csv_file)
    dataset_aimachine_pt = get_aug_sibright_lesions(root=dataset_params['root_path'],
                                                    csv_file=csv_file,
                                                    width=dataset_params['width'],
                                                    csv_split_size=data_params['csv_split_size'],
                                                    image_type=dataset_params['image_type'],
                                                    transforms=transforms_pt,
                                                    preprocessing=preprocessing_transforms,
                                                    use_db=data_params['use_db'])

    dataset_horizonms_pt = SibrightLesionsClassificationPng(root=dataset_params['root_path'],
                                                            csv_file=csv_file,
                                                            width=dataset_params['width'],
                                                            csv_split_size=data_params['csv_split_size'],
                                                            nb_bits=8,
                                                            transforms_pt=transforms_pt,
                                                            transforms_cv=None,
                                                            to_tensor=True)

    dataset_horizonms_cv = SibrightLesionsClassificationPng(root=dataset_params['root_path'],
                                                            csv_file=csv_file,
                                                            width=dataset_params['width'],
                                                            csv_split_size=data_params['csv_split_size'],
                                                            nb_bits=8,
                                                            transforms_pt=None,
                                                            transforms_cv=transforms_cv,
                                                            to_tensor=True)

    dataset_horizonms_pt_cv = SibrightLesionsClassificationPng(root=dataset_params['root_path'],
                                                               csv_file=csv_file,
                                                               width=dataset_params['width'],
                                                               csv_split_size=data_params['csv_split_size'],
                                                               nb_bits=8,
                                                               transforms_pt=transforms_pt,
                                                               transforms_cv=transforms_cv,
                                                               to_tensor=True)

    test_combination = {'aimachine_pt_VS_horizonms_pt': [dataset_aimachine_pt, dataset_horizonms_pt],
                        'aimachine_pt_VS_horizonms_cv': [dataset_aimachine_pt, dataset_horizonms_cv],
                        }
    ## 通过这个对比组合可以得出:
    ## 1.horizonms的dataset和aimachine的dataset经过扩增生成的数据完全一致(不管是pytorch还是cv).
    ## 2.horizonms的dataset经过pytorch扩增和经过cv扩增生成的数据完全一致.

    for key in test_combination.keys():
        print(key)
        for index in range(20):
            print(f"Comparing image {index} ...")
            seed = np.random.randint(0, 100)
            torch.manual_seed(seed)
            image_aim, target_aim = test_combination[key][0].__getitem__(index)
            torch.manual_seed(seed)
            image_hms, target_hms = test_combination[key][1].__getitem__(index)
            # print(f"max = {image_hms.max()}, min = {image_hms.min()}")
            # print(image_aim.shape, image_hms.shape)
            # print(type(image_aim), type(image_hms))
            print(torch.sum(torch.abs(image_hms - image_aim)))
            assert torch.allclose(image_aim.float(), image_hms.float())
            for v_aim, v_hms in zip(target_aim['labels'].value, target_hms['labels'].value):
                assert torch.equal(v_aim, v_hms)
                # print(v_hms)
    # 因为上述代码已经确定horizonms的ptorch和cv的扩增一致,因此混合扩增仅需要测试跑通即可(注意,混合扩增的时候注意扩增方法不要在两个方法里面重复,例如normalizer仅应该出现在pytorch后,因为pytorch最后运行
    for index in range(20):
        print(f"Comparing image {index} ...")
        seed = np.random.randint(0, 100)
        torch.manual_seed(seed)
        image_aim, target_aim = dataset_horizonms_pt_cv.__getitem__(index)
