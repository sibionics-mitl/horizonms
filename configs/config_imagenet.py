import os

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

def get_experiment_config(net_name, scheduler='OneCycleLR'):

    netSize_dict = {'resnet50': dict(width=224, batch_size=64, worker=4),
                    'vgg19_bn': dict(width=224, batch_size=32, worker=4)}
    netparams_dict= netSize_dict[net_name]

    _C = {}
    _C['device'] = 'cuda'
    _C['workers'] = netparams_dict['worker']

    # -----------------------------------------------------------------------------
    # dataset
    # -----------------------------------------------------------------------------
    dataset = dict(root_path='data/imagenet', name='imagenet', image_type='float')
    _C['dataset'] = dataset

    # -----------------------------------------------------------------------------
    # data parameters
    # -----------------------------------------------------------------------------
    data_params = {}
    # data_params['normalizer'] = dict(mode='customize', shift=[0.485*255, 0.456*255, 0.406*255],
    #                                  scale=[0.229*255, 0.224*255, 0.225*255])
    data_params['normalizer'] = dict(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    data_params['color_jitter'] = dict(brightness=0.4, contrast=0.4, saturation=0.4)
    data_params['lighting'] = dict(alphastd=0.1, eigval=[0.2175, 0.0188, 0.0045],
                                    eigvec=[[-0.5675,  0.7192,  0.4009],
                                            [-0.5808, -0.0045, -0.8140],
                                            [-0.5836, -0.6948,  0.4203]])
    data_params['size'] = 224
    data_params['size_enlarge'] = 256
    _C['data_params'] = data_params

    # -----------------------------------------------------------------------------
    # batch processing parameters
    # -----------------------------------------------------------------------------
    batch_params = {}
    batch_params['smoothing'] = 0.1
    batch_params['alpha'] = 0.2
    _C['batch_params'] = batch_params

    # -----------------------------------------------------------------------------
    # network parameters
    # -----------------------------------------------------------------------------
    net_params = {}
    net_params['net_name'] = net_name
    net_params['input_dim'] = 3
    net_params['num_classes'] = 1000
    if 'efficient' in net_name:
        net_params['pretrained'] = ''
        net_params['model_dir'] = 'results'
        net_params['softmax'] = True
    elif net_name == 'InceptionResnetV2':
        net_params['drop_rate'] = 0.2
    # net_params['priors'] = [0, -2.4, -1.6, -3.5, -3.5] # approximately proportional to the ratio of each class
    _C['net_params'] = net_params

    _C['losses'] = [('SoftmaxCrossEntropyLoss', {'mode': 'all'}, 1)]
    _C['metrics'] = [('AccuracyTopk', {'k': 5}),
                     ('Accuracy', {})]

    # -----------------------------------------------------------------------------
    # model training
    # -----------------------------------------------------------------------------
    train_params = {}
    train_params['batch_size'] = netparams_dict['batch_size']
    train_params['epoch_per_save'] = 1
    train_params['lr'] = 0.1
    train_params['clipnorm'] = None#0.001
    train_params['optimizer'] = 'SGD'
    # adam
    train_params['betas'] = (0.9, 0.999)
    # sgd
    train_params['momentum'] = 0.9
    train_params['weight_decay'] = 1e-4#3e-6
    train_params['no_bias_decay'] = 1e-4#3e-6
    if scheduler == 'MultiStepLR':
        # MultiStepLR scheduler
        train_params['epochs'] = 100
        train_params['lr_milestones'] = [30, 60, 90] #epochs = 100
        train_params['lr_gamma'] = 0.1
        train_params['lr_scheduler_mode'] = 'epoch'
        train_params['lr_scheduler_metric'] = False
        train_params['warmup_epochs'] = 0
    elif scheduler == 'OneCycleLR':
        # OneCycleLR scheduler
        train_params['epochs'] = 20
        train_params['max_lr'] = 2.0
        train_params['pct_start'] = 0.45
        train_params['anneal_strategy'] = 'cos'#'linear'
        train_params['cycle_momentum'] = True
        train_params['div_factor'] = 20.0
        train_params['final_div_factor'] = 1000.0
        train_params['three_phase'] = True
        train_params['lr_scheduler_mode'] = 'iteration'
        train_params['lr_scheduler_metric'] = False
        train_params['warmup_epochs'] = 0
    # print
    train_params['print_freq'] = 100
    # validation and model  saving params
    train_params['epoch_save_start'] = 0
    train_params['sub_epoch'] = 1
    # checkpoints
    train_params['save_checkpoints'] = [dict(name='save_all', mode='all')]
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
