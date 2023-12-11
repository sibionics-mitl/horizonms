import os

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
def get_experiment_config(optimizer='Adam', scheduler='ReduceLROnPlateau'):
    _C = {}
    _C['device'] = 'cuda'

    # -----------------------------------------------------------------------------
    # dataset
    # -----------------------------------------------------------------------------
    dataset = {}
    dataset['name'] = 'promise'
    dataset['root_path'] = 'data/prostate/PROSTATE-Aug'
    dataset['train_path'] = ('train/img', 'train/gt')
    dataset['valid_path'] = ('val/img', 'val/gt')
    dataset['grp_regex']  = '(\\d+_Case\\d+_\\d+)_\\d+'
    _C['dataset_params'] = dataset

    # -----------------------------------------------------------------------------
    # data parameters
    # -----------------------------------------------------------------------------
    data_params = {}
    data_params['workers'] = 0#4
    data_params['transforms'] = dict(name='Normalizer', mode='zscore')
    _C['data_params'] = data_params

    # -----------------------------------------------------------------------------
    # network parameters
    # -----------------------------------------------------------------------------
    net_params = dict(
        name='ResidualUNet',
        input_dim=1,
        num_classes=1,
    )
    losses = [dict(name='MILUnarySigmoidLoss', mode='all', focal_params=dict(alpha=0.25, gamma=2.0, sampling_prob=1.0), loss_weight=1.0),
              dict(name='MILPairwiseLoss', softmax=False, exp_coef=-1, loss_weight=10.0)]
    _C['model_params'] = dict(net_params=net_params, loss_params=losses)

    # -----------------------------------------------------------------------------
    # model training
    # -----------------------------------------------------------------------------
    train_params = {}
    train_params['batch_size'] = 16
    train_params['epochs'] = 50
    train_params['epoch_per_save'] = 1
    train_params['start_epoch'] = 0
    train_params['lr'] = 1e-4
    train_params['clipnorm'] = 0.001
    train_params['optimizer'] = optimizer
    if optimizer == 'Adam':
        train_params['betas'] = (0.9, 0.999)
    else:
        train_params['momentum'] = 0.9
        train_params['weight_decay'] = 1e-4
    train_params['scheduler'] = scheduler
    if scheduler == 'MultiStepLR':
        # MultiStepLR scheduler
        train_params['lr_milestones'] = [55, 70]
        train_params['lr_gamma'] = 0.1
        train_params['lr_scheduler_mode'] = 'epoch'
        train_params['lr_scheduler_metric'] = False
    elif scheduler == 'ReduceLROnPlateau':
        # ReduceLROnPlateau scheduler
        train_params['factor'] = 0.5
        train_params['patience'] = 8
        train_params['lr_scheduler_mode'] = 'epoch'
        train_params['lr_scheduler_metric'] = True
    train_params['warmup_epochs'] = 0
    
    # # print
    train_params['print_freq'] = 50
    # validation and model saving params
    train_params['epoch_save_start'] = 0
    train_params['sub_epoch'] = 1
    # checkpoints
    train_params['save_checkpoints'] = [dict(name='save_all', mode='all')]


    _C['train_params'] = train_params


    # -----------------------------------------------------------------------------
    # model saving
    # -----------------------------------------------------------------------------
    save_params = {}
    save_params['dir_save_root']   = 'results'
    save_params['dir_save']        = os.path.join('results',dataset['name'])
    save_params['experiment_name'] = None
    _C['save_params'] = save_params

    return _C
