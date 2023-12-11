import os
import cv2

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

def get_experiment_config(net_name='YOLOv1', scheduler='MultiStepLR', optimizer='SGD'):

    _C = {}
    _C['device'] = 'cuda'
    _C['workers'] = 4

    # -----------------------------------------------------------------------------
    # dataset
    # -----------------------------------------------------------------------------
    dataset = dict(
                root_path='/data1/home/panxc/wsis/data/voc',
                name='voc', image_type='float', keep_difficult=False,
                train_years=['2012'], test_years=['2012'],
                # train_sets=["train", "train"], #
                train_sets=["train"], 
                test_sets=['val'])
    _C['dataset_params'] = dataset

    # -----------------------------------------------------------------------------
    # data parameters
    # -----------------------------------------------------------------------------
    data_params = {}
    data_params['transforms_train'] = dict(
        name='SequentialAugment', 
        augment_operators = [
            dict(name='CVRandomFliplr', prob=0.5),
            dict(name='CVRandomScale', prob=0.5, scale_range=[0.8, 1.2], 
                 scale_width=True, scale_height=False, scale_same=False), 
            dict(name='CVRandomGaussianBlur', prob=0.5, sigma=0.3, kernel_size=(5,5)),
            dict(name='CVRandomBrightness', prob=0.5, brightness_factor=[0.5,1.5]),
            dict(name='CVRandomHue', prob=0.5, hue_factor=[0.5,1.5]),
            dict(name='CVRandomSaturation', prob=0.5, saturation_factor=[0.5,1.5]),
            dict(name='CVRandomShift', prob=0.5, shift_limit=0.2),
            dict(name='CVRandomCrop', prob=0.5, crop_ratio=0.6),
            dict(name='CVNormalizer', mode='customize', shift=(123,117,104), scale=[1,1,1]),
            dict(name='CVResize', size=(448, 448))
            ]
        )
    data_params['transforms_test'] = dict(
        name='SequentialAugment', 
        augment_operators = [
            dict(name='CVNormalizer', mode='customize', shift=(123,117,104), scale=[1,1,1]),
            dict(name='CVResize', size=(448, 448))
            ]
        )
    _C['data_params'] = data_params

    # -----------------------------------------------------------------------------
    # network parameters for YOLO
    # -----------------------------------------------------------------------------
    model_params = dict(name='YOLODetection')
    net_params = {}
    net_params['name'] = net_name
    if net_name == 'YOLOv1':
        backbone = dict(name='resnet_backbone', backbone_name='resnet50',
            input_dim=3, return_stages=1,
            #   pretrained=True
             pretrained='imagenet'
            )
        neck = dict(name='BottlenetNeck', in_channels=2048, out_channels=256,
            stride=1, expansion=1)
        head = dict(name='Yolov1Head', in_channels=256, num_classes=20, num_boxes=2, prior=0.5)
        net_params['backbone'] = backbone
        net_params['neck'] = neck
        net_params['head'] = head
        net_params['stride'] = 32
        model_params['loss_params'] = dict(name='YOLOv1Losses', lambda_coord=5, lambda_obj=1, lambda_noobj=0.5)
        model_params['metric_params'] = dict(name='YOLOv1Metrics')
    model_params['net_params'] = net_params
    _C['model_params'] = model_params
    # _C['net_params'] = net_params

    # -----------------------------------------------------------------------------
    # model training
    # -----------------------------------------------------------------------------
    train_params = {}
    train_params['batch_size'] = 24
    train_params['epochs'] = 50
    train_params['epoch_per_save'] = 1
    train_params['start_epoch'] = 0
    train_params['lr'] = 1e-3
    train_params['clipnorm'] = None
    train_params['optimizer'] = optimizer
    if optimizer == 'AdamW':
        # adamw
        train_params['betas'] = (0.9, 0.999)
    elif optimizer == 'SGD':
        # sgd
        train_params['momentum'] = 0.9
    train_params['weight_decay'] = 5e-4
    train_params['no_bias_decay'] = None
    if scheduler == 'MultiStepLR':
        # MultiStepLR scheduler
        train_params['lr_milestones'] = [30, 40]
        train_params['lr_gamma'] = 0.1
        train_params['lr_scheduler_mode'] = 'epoch'
        train_params['lr_scheduler_metric'] = False
        train_params['warmup_epochs'] = 0
    elif scheduler == 'ReduceLROnPlateau':
        # ReduceLROnPlateau scheduler
        train_params['factor'] = 0.5
        train_params['patience'] = 4
        train_params['lr_scheduler_mode'] = 'epoch'
        train_params['lr_scheduler_metric'] = True
        train_params['warmup_epochs'] = 2
        train_params['early_stop_patience'] = 0
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
    train_params['print_freq'] = 100
    # validation and model saving params
    train_params['epoch_save_start'] = 10
    train_params['sub_epoch'] = 1
    # checkpoints
    train_params['save_checkpoints'] = [dict(name='save_all', mode='all')]
    _C['train_params'] = train_params

    # -----------------------------------------------------------------------------
    # model saving
    # -----------------------------------------------------------------------------
    save_params = {}
    save_params['dir_save_root']   = 'results'
    save_params['dir_save']        = os.path.join('results', dataset['name'])
    save_params['experiment_name'] = None
    _C['save_params'] = save_params

    return _C
