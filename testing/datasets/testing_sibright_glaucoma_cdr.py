import os, sys
sys.path.insert(0, os.getcwd())
import torch
from aimachine import transforms as T
from aimachine.datasets_utils.sibright_glaucoma_cdr import get_aug_sibright_glaucoma_cdr, get_aug_sibright_glaucoma_cdr_detection

from horizonms.datasets import SibrightGlaucomaCDRSegmentationPng, SibrightGlaucomaCDRDetectionPng


if __name__ == "__main__":
    task = 'segmentation'
    task = 'detection'
    if task == 'segmentation':
        from configs.config_sibright_glaucoma_cdr import get_experiment_config
        exp_config = get_experiment_config()
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
                transforms.append(T.RandomCrop(**data_params['random_crop']))
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
        dataset_aimachine = get_aug_sibright_glaucoma_cdr(root=dataset_params['root_path'], 
                                    csv_file=csv_file,
                                    folder=dataset_params['png_path'],
                                    width=dataset_params['width'], 
                                    softmax=exp_config['softmax'],
                                    image_type=dataset_params['image_type'],
                                    transforms=transforms,
                                    preprocessing=preprocessing_transforms,
                                    use_db=data_params['use_db'])


        dataset_horizonms = SibrightGlaucomaCDRSegmentationPng(
                    root=dataset_params['root_path'], 
                    csv_file=csv_file,
                    folder=dataset_params['png_path'],
                    width=dataset_params['width'], 
                    softmax=exp_config['softmax'],
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

            d = (image_aim - image_hms).abs().sum()
            print(d)
            assert torch.allclose(image_aim, image_hms)
            d = (target_aim['masks'].value - target_hms['masks'].value).abs().sum()
            print(d)
            assert torch.allclose(target_aim['masks'].value, target_hms['masks'].value)
    else:
        from configs.config_sibright_glaucoma_cdr_detection import get_experiment_config
        net_name = 'Retinanet'
        exp_config = get_experiment_config(net_name)
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
                transforms.append(T.RandomCrop(**data_params['random_crop']))
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
        dataset_aimachine = get_aug_sibright_glaucoma_cdr_detection(
                                root=dataset_params['root_path'], 
                                csv_file=csv_file,
                                folder=dataset_params['png_path'],
                                width=dataset_params['width'], 
                                image_type=dataset_params['image_type'],
                                transforms=transforms,
                                preprocessing=preprocessing_transforms,
                                use_db=data_params['use_db'])



        dataset_horizonms = SibrightGlaucomaCDRDetectionPng(
                    root=dataset_params['root_path'], 
                    csv_file=csv_file,
                    folder=dataset_params['png_path'],
                    width=dataset_params['width'], 
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

            d = (image_aim - image_hms).abs().sum()
            print("image: ", d)
            assert torch.allclose(image_aim, image_hms)

            d = (target_aim['masks'].value - target_hms['masks'].value).abs().sum()
            print("masks: ", d)
            assert torch.allclose(target_aim['masks'].value, target_hms['masks'].value)

            d = (target_aim['bboxes'].value - target_hms['bboxes'].value).abs().sum()
            print("bboxes: ", d)
            assert torch.allclose(target_aim['bboxes'].value.float(), target_hms['bboxes'].value)
