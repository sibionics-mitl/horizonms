import os

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

def get_experiment_config(net_name, scheduler='OneCycleLR', optimizer='AdamW'):

    netSize_dict = {'efficientnet_b3': dict(width=300, batch_size=32, worker=8),
                    'efficientnet_b4': dict(width=380, batch_size=24, worker=8),
                    'efficientnet_b5': dict(width=456, batch_size=8, worker=8)}
    netparams_dict= netSize_dict[net_name]

    _C = {}
    _C['device'] = 'cuda'
    _C['workers'] = netparams_dict['worker']

    # -----------------------------------------------------------------------------
    # dataset
    # -----------------------------------------------------------------------------
    dataset = dict(root_path='data/EyePACs', name='eyepacs', image_type='float')
    _C['dataset'] = dataset

    # -----------------------------------------------------------------------------
    # data parameters
    # -----------------------------------------------------------------------------
    data_params = {}
    data_params['image_scale_factor'] = 1.0
    data_params['normalizer_mode'] = 'zscore'
    data_params['fold_id'] = 1
    data_params['valid_seed'] = 0
    data_params['resize'] = dict(width=netparams_dict['width'])
    data_params['use_db'] = 'png'
    data_params['trivalaugment'] = dict(
            augment_operators=[dict(type='Identity'),
                               dict(type='Brightness', params=dict(param_range=[0.5, 2.0])),
                               dict(type='Contrast', params=dict(param_range=[0.5, 2.0])),
                               dict(type='Saturation', params=dict(param_range=[0.5, 2.0])),
                               dict(type='Sharpness', params=dict(param_range=[0.5, 2.0])),
                            #    dict(type='Hue', params=dict(param_range=[-0.1, 0.1])), #
                            #    dict(type='Posterize', params=dict(param_range=[2, 8])), #
                            #    dict(type='Solarize', params=dict(param_range=[1.0, 0.85])), #
                            #    dict(type='AutoContrast'),
                            #    dict(type='Equalize'), #
                            #    dict(type='GaussianBlur', params=dict(param_range=[0.1, 1.2], kernel_size=7)),
                            #    dict(type='GaussianNoise', params=dict(param_range=[-0.1, 0.1], mean=0)),
                               dict(type='TranslateX', params=dict(param_range=[-0.05, 0.05])), #
                               dict(type='TranslateY', params=dict(param_range=[-0.025, 0.025])), #
                               dict(type='CropX', params=dict(param_range=[-0.1, 0.1])),
                               dict(type='CropY', params=dict(param_range=[-0.05, 0.05])),
                               dict(type='Fliplr'), dict(type='Flipud'),
                               dict(type='Rotate', params=dict(param_range=[-90, 90])),
                               ],
            num_magnitude_bins=32)
    _C['data_params'] = data_params

    # -----------------------------------------------------------------------------
    # network parameters
    # -----------------------------------------------------------------------------
    net_params = {}
    net_params['net_name'] = net_name
    net_params['input_dim'] = 3
    net_params['num_classes'] = 5
    if 'efficient' in net_name:
        net_params['pretrained'] = ''
        net_params['model_dir'] = 'results'
    elif net_name == 'InceptionResnetV2':
        net_params['drop_rate'] = 0.2
    # net_params['priors'] = [0, -2.4, -1.6, -3.5, -3.5] # approximately proportional to the ratio of each class
    _C['net_params'] = net_params
    _C['softmax'] = True
    _C['losses'] = [('SoftmaxCrossEntropyLoss', {'mode': 'all'}, 1)]
    _C['metrics'] = [('SoftmaxCohenKappaScore', {'weights': 'quadratic', 'category': True}),
                     ('SoftmaxAccuracy', {})]

    # -----------------------------------------------------------------------------
    # model training
    # -----------------------------------------------------------------------------
    train_params = {}
    train_params['batch_size'] = netparams_dict['batch_size']
    train_params['epochs'] = 120
    train_params['epoch_per_save'] = 1
    train_params['start_epoch'] = 0
    # train_params['resume'] = ''
    # train_params['test_only'] = False
    train_params['lr'] = 1e-3 * (train_params['batch_size'] / 32.0)
    train_params['clipnorm'] = 0.001
    train_params['optimizer'] = optimizer
    if optimizer == 'AdamW':
        # adamw
        train_params['betas'] = (0.9, 0.999)
    else:
        # sgd
        train_params['momentum'] = 0.9
    train_params['weight_decay'] = 0
    train_params['no_bias_decay'] = None
    if scheduler == 'MultiStepLR':
        # MultiStepLR scheduler
        train_params['lr_milestones'] = [55, 70]
        train_params['lr_gamma'] = 0.1
        train_params['lr_scheduler_mode'] = 'epoch'
        train_params['lr_scheduler_metric'] = False
        train_params['warmup_epochs'] = 2
    elif scheduler == 'ReduceLROnPlateau':
        # ReduceLROnPlateau scheduler
        train_params['factor'] = 0.5
        train_params['patience'] = 2
        train_params['lr_scheduler_mode'] = 'epoch'
        train_params['lr_scheduler_metric'] = True
        train_params['warmup_epochs'] = 2
    elif scheduler == 'StepLR':
        # StepLR scheduler
        train_params['step_size'] = 5
        train_params['step_gamma'] = 0.8
        train_params['lr_scheduler_mode'] = 'epoch'
        train_params['lr_scheduler_metric'] = False
        train_params['warmup_epochs'] = 2
    elif scheduler == 'CyclicLR':
        train_params['base_lr'] = 0.0001
        train_params['max_lr'] = 0.01
        train_params['step_size_up'] = train_params['epochs'] // 2
        train_params['lr_scheduler_mode'] = 'epoch'
        train_params['lr_scheduler_metric'] = False
        train_params['warmup_epochs'] = 0
    elif scheduler == 'OneCycleLR':
        # OneCycleLR scheduler
        train_params['max_lr'] = 0.01
        train_params['pct_start'] = 0.5
        train_params['anneal_strategy'] = 'linear'
        train_params['cycle_momentum'] = False
        train_params['div_factor'] = 100.0
        train_params['final_div_factor'] = 1000.0
        train_params['three_phase'] = False
        train_params['lr_scheduler_mode'] = 'iteration'
        train_params['lr_scheduler_metric'] = False
        train_params['warmup_epochs'] = 0
    # print
    train_params['print_freq'] = 200
    # validation and model saving params
    train_params['epoch_save_start'] = 20
    train_params['sub_epoch'] = 1
    # checkpoints
    train_params['save_checkpoints'] = [dict(name='val_loss', mode='min'),
                                        dict(name='val_SoftmaxCohenKappaScore', mode='max'),
                                        dict(name='val_SoftmaxAccuracy', mode='max')]
    _C['train_params'] = train_params


    # -----------------------------------------------------------------------------
    # model saving
    # -----------------------------------------------------------------------------
    save_params = dict(
        dir_save_root = 'result',
        dir_save = os.path.join('results', dataset['name']),
        experiment_name = None)
    _C['save_params'] = save_params

    return _C
