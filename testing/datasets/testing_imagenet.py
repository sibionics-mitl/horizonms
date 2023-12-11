import os, sys
sys.path.insert(0, os.getcwd())
import torch
import torchvision.transforms as tv_transforms
from configs.config_imagenet import get_experiment_config 
from aimachine import transforms as T
from aimachine.datasets_utils.imagenet import get_aug_imagenet

from horizonms.datasets import ImageNetClassification


if __name__ == "__main__":
    net_name = 'resnet50'
    scheduler = 'MultiStepLR'
    exp_config = get_experiment_config(net_name, scheduler)
    data_params = exp_config['data_params']
    dataset_params = exp_config['dataset']

    train_transforms_aim = [tv_transforms.RandomResizedCrop(data_params['size']),
                        tv_transforms.RandomHorizontalFlip (p=0.5),
                        tv_transforms.Normalize(**data_params['normalizer'])]
    train_transforms_hms = [T.RandomResizedCrop(data_params['size']),
                        # T.RandomHorizontalFlip (p=0.5),
                        # T.Normalizer(**data_params['normalizer'])
                        ]
    val_transforms_aim = [#T.Uint8ToFloat(),
                      tv_transforms.Resize(data_params['size_enlarge']),
                      tv_transforms.CenterCrop(data_params['size']),
                      tv_transforms.Normalize(**data_params['normalizer'])]
    val_transforms_hms = [
                        # T.Uint8ToFloat(),
                        T.Resize(data_params['size_enlarge']),
                        # T.CenterCrop(data_params['size']),
                        # T.Normalizer(**data_params['normalizer'])
                        ]



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
    mode = "val"
    if mode == 'train':
        transforms_aim = train_transforms_aim
        transforms_hms = train_transforms_hms
    else:
        transforms_aim = val_transforms_aim
        transforms_hms = val_transforms_hms
    print(transforms_hms)
    dataset_aimachine = get_aug_imagenet(
                            root=dataset_params['root_path'], 
                            mode=mode, 
                            image_type=dataset_params['image_type'],
                            transforms=transforms_aim)


    dataset_horizonms = ImageNetClassification(
                root=dataset_params['root_path'], 
                mode=mode,
                nb_bits=8,
                transforms_pt=transforms_hms,
                transforms_cv=None,
                to_tensor=True) 
    
    for index in range(20):
        print(f"Comparing image {index} ...")
        image_aim, target_aim = dataset_aimachine.__getitem__(index)
        print(target_aim['labels'].value)
        image_hms, target_hms = dataset_horizonms.__getitem__(index)
        # # print(f"max = {image_hms.max()}, min = {image_hms.min()}")
        # # print(image_aim.shape, image_hms.shape)

        assert torch.allclose(image_aim, image_hms)
        assert torch.equal(target_aim['labels'].value, target_hms['labels'].value)
        # print(target_hms['labels'].value)
