import os, sys
sys.path.insert(0, os.getcwd())
import numpy as np
import torch
from horizonms.models.classification import Classification
from horizonms import transforms as T
from horizonms.datasets import SibrightDiseasesClassificationPng
from aimachine.engine import utils
from configs.config_sibright_diseases import get_experiment_config
torch.multiprocessing.set_sharing_strategy('file_system')


if __name__ == "__main__":    
    net_name = 'efficientnet_mixmultiheads_b3'
    scheduler = 'OneCycleLR'
    exp_config = get_experiment_config(net_name, scheduler)
    data_params = exp_config['data_params']
    dataset_params = exp_config['dataset']
    device = exp_config['device']
    model = Classification(exp_config['net_params'], final_activation=None, 
                           loss_params=exp_config['loss_params'], 
                           metric_params=exp_config['metric_params'])
    model.to(device)

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

    csv_file = dataset_params['train_csv']
    transforms = get_transform(True)
    dataset = SibrightDiseasesClassificationPng(
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
    image, target = dataset.__getitem__(0)
    # print(image.shape, target)
    # print(target['labels'].value)
    
    train_sampler = torch.utils.data.RandomSampler(dataset)
    train_batch_sampler = torch.utils.data.BatchSampler(
            train_sampler, 8, drop_last=True)
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_sampler=train_batch_sampler, num_workers=exp_config['workers'],
        collate_fn=utils.collate_fn, pin_memory=False)

    iter = 0
    non_blocking = True
    model.train()
    for images, targets in train_loader:
        images = list(image.to(device, non_blocking=non_blocking) for image in images)
        targets = [{k: v.to(device, non_blocking=non_blocking) for k, v in t.items() 
                    if not isinstance(v.value, str)} for t in targets]
        
        loss, pred = model.forward_train(images, targets)
        print(loss)
        print([v.shape for v in pred])
        losses, metrics, ypred = model.test_one_batch(images, targets)
        print(metrics)
        break

    