import torch
from functools import partial
import numpy as np
from .engine.optimizer import add_weight_decay, AGC
from .engine import Trainer, CheckpointMetric, utils
from .samplers import BalancedSampler, DistributedBalancedSampler


def create_dataloader(dataset_class, dataset_params_common, 
                      dataset_params_train, dataset_params_valid=None, sample_weighting=None,
                      batch_size=8, valid_batch_size=None, workers=0, distributed=False):
    # Data loading code
    print("Loading data")
    
    sibright_dataset = partial(dataset_class, **dataset_params_common)
    dataset = sibright_dataset(**dataset_params_train)
    if dataset_params_valid is not None:
        dataset_test = sibright_dataset(**dataset_params_valid)
        print(f"#train = {len(dataset)}, #test = {len(dataset_test)}")
    else:
        print(f"#train = {len(dataset)}, no testing data is provided.")

    print("Creating data loaders")
    if sample_weighting is not None:
        sampling_labels = np.array(dataset.sampling_labels)
        nb_samples = np.array([np.sum(sampling_labels==k) for k in range(dataset.sampling_classes)])
        if sample_weighting == 'square_root':
            nb_samples_per_class = np.sqrt(nb_samples)
        elif sample_weighting == 'equal':
            nb_samples_per_class = np.array([1] * dataset.sampling_classes)
        nb_samples_per_class = np.around(nb_samples_per_class / nb_samples_per_class.sum() * nb_samples.sum()).astype(np.int32)
        nb_samples_per_class = list(nb_samples_per_class)
        print(f"Sampling: nb_samples_per_class = {nb_samples_per_class}")
    if distributed:
        if sample_weighting is None:
            train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            train_sampler = DistributedBalancedSampler(dataset,
                                samples_per_class=nb_samples_per_class)
        if dataset_params_valid is not None:
            test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        if sample_weighting is None:
            train_sampler = torch.utils.data.RandomSampler(dataset)
        else:
            train_sampler = BalancedSampler(dataset, samples_per_class=nb_samples_per_class)
        if dataset_params_valid is not None:
            test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    train_batch_sampler = torch.utils.data.BatchSampler(
            train_sampler, batch_size, drop_last=True)

    pin_memory = True
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_sampler=train_batch_sampler, num_workers=workers,
        collate_fn=utils.collate_fn, pin_memory=pin_memory)

    if dataset_params_valid is not None:
        if valid_batch_size is None:
            valid_batch_size = 2 * batch_size
        valid_loader = torch.utils.data.DataLoader(
            dataset_test, batch_size=valid_batch_size,
            sampler=test_sampler, num_workers=2*workers,
            collate_fn=utils.collate_fn, pin_memory=pin_memory)
    else:
        valid_loader = None
    return train_loader, valid_loader, train_sampler


def training(train_loader, valid_loader, model, train_params, 
             scheduler='ReduceLROnPlateau', train_sampler=None,
             distributed=False, output_dir='./'):
    nb_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('# trainable parameters: {}'.format(nb_params))
    no_bias_decay = train_params.get('no_bias_decay', None)
    if no_bias_decay is not None:
        print(f"no_bias_decay = {no_bias_decay}")
        params = add_weight_decay(model, no_bias_decay)
    else:
        params = [p for p in model.parameters() if p.requires_grad]
    if train_params['optimizer'] == 'SGD':
        if train_params['no_bias_decay'] is not None:
            optimizer = torch.optim.SGD(params, lr=train_params['lr'], 
                                        momentum=train_params['momentum'])
        else:
            optimizer = torch.optim.SGD(params, lr=train_params['lr'], 
                                        momentum=train_params['momentum'], 
                                        weight_decay=train_params['weight_decay'])
    elif train_params['optimizer'] == 'AdamW':
        if train_params['no_bias_decay'] is not None:
            print(f"adamw with no_bias_decay = {train_params['no_bias_decay']}")
            optimizer = torch.optim.AdamW(params, lr=train_params['lr'],
                                         betas=train_params['betas'])
        else:
            print(f"adamw with weight_decay = {train_params['weight_decay']}")
            optimizer = torch.optim.AdamW(params, lr=train_params['lr'],
                                         betas=train_params['betas'],
                                         weight_decay=train_params['weight_decay'])
    elif train_params['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(params, lr=train_params['lr'], betas=train_params['betas'])
    
    if train_params.get('use_agc',False):
        linear_layers_list= []
        for name ,_ in model.named_modules() :
            if 'fc' in name or 'linear' in name or 'classifiers' in name :
                linear_layers_list.append(name)
        #print(linear_layers_list)
        optimizer = AGC(model.parameters(), optimizer, model=model, ignore_agc=linear_layers_list)

    save_checkpoints = [CheckpointMetric(**param) for param in train_params['save_checkpoints']]
                 
    trainer = Trainer(model, optimizer, train_loader, valid_loader,
                      lr_scheduler_mode=train_params['lr_scheduler_mode'],
                      warmup_epochs=train_params['warmup_epochs'],
                      distributed=distributed, train_sampler=train_sampler,
                      epochs=train_params['epochs'], non_blocking=True,
                      output_dir=output_dir)
    
    if scheduler == 'MultiStepLR':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                    milestones=train_params['lr_milestones'], gamma=train_params['lr_gamma'])
    elif scheduler == 'ReduceLROnPlateau':
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                    factor=train_params['factor'], patience=train_params['patience'])
    elif scheduler == 'StepLR':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                    step_size=train_params['step_size'], gamma=train_params['step_gamma'])
    elif scheduler == 'CyclicLR':
        lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, 
                    base_lr=train_params['base_lr'], max_lr=train_params['max_lr'],
                    step_size_up=train_params['step_size_up'], mode='triangular',
                    cycle_momentum=False)
    elif scheduler == 'OneCycleLR':
        max_lr = train_params['max_lr']
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, 
                                            epochs=train_params['epochs'],
                                            steps_per_epoch=len(train_loader),
                                            pct_start=train_params['pct_start'], 
                                            anneal_strategy=train_params['anneal_strategy'] ,
                                            cycle_momentum=train_params['cycle_momentum'],
                                            div_factor=train_params['div_factor'],
                                            final_div_factor=train_params['final_div_factor'],
                                            three_phase=train_params['three_phase'])
    
    trainer.set_lr_scheduler(lr_scheduler, lr_scheduler_metric=train_params['lr_scheduler_metric'])
    trainer.train(sub_epoch=train_params['sub_epoch'], 
                 epoch_per_save=train_params['epoch_per_save'],
                 epoch_save_start=train_params['epoch_save_start'],
                 save_checkpoints=save_checkpoints,
                 clipnorm=train_params['clipnorm'],
                 print_freq=train_params['print_freq'])