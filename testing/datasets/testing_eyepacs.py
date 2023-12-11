import os, sys
sys.path.insert(0, os.getcwd())
import torch
from configs.config_eyepacs import get_experiment_config 
from aimachine import transforms as T
from aimachine.datasets_utils.eyepacs import get_aug_eyepacs

from horizonms.datasets import EyePACSClassificationPng


if __name__ == "__main__":
    net_name = 'efficientnet_b5'
    scheduler = 'OneCycleLR'
    exp_config = get_experiment_config(net_name, scheduler)
    data_params = exp_config['data_params']
    dataset_params = exp_config['dataset']

    def get_transform(mode='train'):
        transforms = []
        if mode == 'train':
            trivalaugment = data_params['trivalaugment']
            augment_operators = []
            for augment in trivalaugment['augment_operators']:
                if augment.get('params', False):
                    augment_operators.append(T.OpStructure(type=augment['type'], value=T.OpParamStructure(**augment['params'])))
                else:
                    augment_operators.append(T.OpStructure(type=augment['type']))
            transforms.append(T.CustomizedTrivialAugment(augment_operators, trivalaugment['num_magnitude_bins']))
        if data_params.get("resize_shape", False):
            print("fixed shape is used for image resize")
            w = data_params['resize']['width']
            transforms.append(T.ResizeShape((w, w)))
        else:
            print("fixed width is used for image resize")
            transforms.append(T.ResizeWidth(**data_params['resize']))
            transforms.append(T.ImageHeightPaddingOrCrop())
        transforms.append(T.Normalizer(mode=data_params['normalizer_mode']))
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
    mode = "valid"
    use_valid = True
    transforms = get_transform(mode)
    print(transforms)
    dataset_aimachine = get_aug_eyepacs(root=dataset_params['root_path'], 
                mode=mode, use_valid=use_valid, 
                image_type=dataset_params['image_type'],
                valid_seed=data_params['valid_seed'],
                transforms=transforms,
                preprocessing=preprocessing_transforms,
                use_db=data_params['use_db'])

    dataset_horizonms = EyePACSClassificationPng(root=dataset_params['root_path'], 
                mode=mode, use_valid=use_valid, 
                valid_seed=data_params['valid_seed'],
                transforms_pt=transforms,
                transforms_cv=None,
                to_tensor=True)
    
    
    for index in range(20):
        print(f"Comparing image {index} ...")
        image_aim, target_aim = dataset_aimachine.__getitem__(index)
        image_hms, target_hms = dataset_horizonms.__getitem__(index)

        assert torch.allclose(image_aim, image_hms)
        assert torch.equal(target_aim['labels'].value, target_hms['labels'].value)
        print(target_hms['labels'].value)
