import os, sys
sys.path.insert(0, os.getcwd())
import torch
import torchvision.transforms as tv_transforms
from configs.config_promise import get_experiment_config 
from aimachine import transforms as T
from aimachine.datasets_utils.promise import get_promise

from horizonms.datasets import PromiseSegmentation


if __name__ == "__main__":
    exp_config = get_experiment_config()
    data_params = exp_config['data_params']
    dataset_params = exp_config['dataset']

    def get_transform_aim():
        transforms = []
        transforms.append(T.ToTensor('float'))
        transforms.append(T.Normalizer(mode=data_params['normalizer_mode']))
        return T.Compose(transforms)

    def get_transform_hms():
        transforms = []
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
    margin = 0
    random = False
    dataset_aimachine = get_promise(root=dataset_params['root_path'], 
                            image_folder=dataset_params['train_path'][0], 
                            gt_folder=dataset_params['train_path'][1], 
                            margin=margin, random=random,
                            transforms=get_transform_aim())


    dataset_horizonms = PromiseSegmentation(
                            root=dataset_params['root_path'], 
                            image_folder=dataset_params['train_path'][0],
                            gt_folder=dataset_params['train_path'][1],
                            margin=margin,
                            random=random,
                            nb_bits=8,
                            transforms_pt=get_transform_hms(),
                            transforms_cv=None,
                            to_tensor=True)
    
    for index in range(20):
        print(f"Comparing image {index} ...")
        image_aim, target_aim = dataset_aimachine.__getitem__(index)
        image_hms, target_hms = dataset_horizonms.__getitem__(index)
        # print(target_aim)
        # print(target_hms)

        assert torch.allclose(image_aim, image_hms)
        for key in list(target_aim.keys()):
            assert torch.equal(target_aim[key].value, target_hms[key].value)
        
