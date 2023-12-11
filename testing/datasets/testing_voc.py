import os, sys
sys.path.insert(0, os.getcwd())
import torch
from configs.config_voc import get_experiment_config
from horizonms import transforms as T
from horizonms.datasets import VOCDetection as VOCDetection, VOCSegmentation


if __name__ == "__main__":
    net_name = 'YOLOv1'
    scheduler = 'MultiStepLR'
    exp_config = get_experiment_config(net_name, scheduler=scheduler)
    data_params = exp_config['data_params']
    dataset_params = exp_config['dataset']

    def get_transform_cv(train):
        transforms = []
        if train:
            print("do augmentation...")
            transforms.append(T.CVRandomFliplr(**data_params['fliplr']))
            transforms.append(T.CVRandomScale(**data_params['scale']))
            transforms.append(T.CVRandomBlur(**data_params['blur']))
            transforms.append(T.CVRandomBrightness(**data_params['brightness']))
            transforms.append(T.CVRandomHue(**data_params['hue']))
            transforms.append(T.CVRandomSaturation(**data_params['saturation']))
            transforms.append(T.CVRandomShift(**data_params['shift']))
            transforms.append(T.CVRandomCrop(**data_params['crop']))
        transforms.append(T.CVNormalizer(**data_params['normalizer']))
        transforms.append(T.CVResize(data_params['size']))
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
    years = dataset_params['test_years']
    image_sets = dataset_params['test_sets']
    transforms_pt = None
    transforms_cv = get_transform_cv(False)
    print("transforms_cv: ", transforms_cv)

    dataset_det = VOCDetection(root=dataset_params['root_path'], 
                years=years,
                image_sets=image_sets,
                download=False,
                keep_difficult=dataset_params['keep_difficult'],
                transforms_pt=None,
                transforms_cv=transforms_cv,
                to_tensor=True)  

    dataset_seg = VOCSegmentation(root=dataset_params['root_path'], 
                years=years,
                image_sets=image_sets,
                download=False,
                keep_difficult=dataset_params['keep_difficult'],
                transforms_pt=None,
                transforms_cv=transforms_cv,
                to_tensor=True)  
    
    for index in range(20):
        print(f"Comparing image {index} for detection ...")
        image_det, target_det = dataset_det.__getitem__(index)

        print(image_det.shape)
        for key, value in target_det.items():
            print(key, target_det[key].value)

        print(f"Comparing image {index} for segmentation ...")
        image_seg, target_seg = dataset_seg.__getitem__(index)

        print(image_seg.shape)
        for key, value in target_seg.items():
            print(key, value.value)
            if value.type == 'masks':
                print(torch.unique(value.value))
