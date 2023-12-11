import os, sys
sys.path.insert(0, os.getcwd())
import torch
from configs.config_sibright_diseases import get_experiment_config 
from aimachine import transforms as T
from aimachine.datasets_utils.sibright_diseases import get_aug_sibright_diseases

from horizonms.datasets import SibrightDiseasesClassificationPng


if __name__ == "__main__":
    net_name = 'efficientnet_mixmultiheads_b3'
    scheduler = 'OneCycleLR'
    exp_config = get_experiment_config(net_name, scheduler)
    data_params = exp_config['data_params']
    dataset_params = exp_config['dataset']

    def get_transform(train: bool):
        transforms = []
        if train:
            trivalaugment = data_params['trivalaugment']
            augment_operators = []
            for augment in trivalaugment['augment_operators']:
                if augment.get('params', False):
                    augment_operators.append(T.OpStructure(type=augment['type'], value=T.OpParamStructure(**augment['params'])))
                else:
                    augment_operators.append(T.OpStructure(type=augment['type']))
            transforms.append(T.CustomizedTrivialAugment(augment_operators, trivalaugment['num_magnitude_bins']))
        
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
    if mode == 'train':
        csv_file = dataset_params['train_csv']
        transforms = get_transform(True)
    else:
        csv_file = dataset_params['valid_csv']
        transforms = get_transform(False)
    print(transforms)
    print(dataset_params['root_path'], csv_file)
    dataset_aimachine = get_aug_sibright_diseases(image_folder=dataset_params['image_folder'], 
                                csv_file=csv_file,
                                width=dataset_params['width'], 
                                csv_split_size=data_params['csv_split_size'],
                                list_multiclass=exp_config['list_multiclass'],
                                list_multilabel=exp_config['list_multilabel'],
                                category_multiclass=exp_config['category_multiclass'],
                                category_multilabel=exp_config['category_multilabel'],
                                image_type=dataset_params['image_type'],
                                transforms=transforms,
                                preprocessing=preprocessing_transforms,
                                use_db=data_params['use_db'])

    dataset_horizonms = SibrightDiseasesClassificationPng(
                image_folder=dataset_params['image_folder'], 
                csv_file=csv_file,
                width=dataset_params['width'], 
                csv_split_size=data_params['csv_split_size'],
                list_multiclass=exp_config['list_multiclass'],
                list_multilabel=exp_config['list_multilabel'],
                category_multiclass=exp_config['category_multiclass'],
                category_multilabel=exp_config['category_multilabel'],
                nb_bits=8,
                transforms_pt=transforms,
                transforms_cv=None,
                to_tensor=True)
    
    for index in range(20):
        print(f"Comparing image {index} ...")
        image_aim, target_aim = dataset_aimachine.__getitem__(index)
        image_hms, target_hms = dataset_horizonms.__getitem__(index)
        # print(f"max = {image_hms.max()}, min = {image_hms.min()}")
        # print(image_aim.shape, image_hms.shape)

        assert torch.allclose(image_aim, image_hms)
        for v_aim, v_hms in zip(target_aim['labels'].value, target_hms['labels'].value):
            assert torch.allclose(v_aim, v_hms)
            # print(v_hms)
