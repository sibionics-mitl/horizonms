import os, sys
sys.path.insert(0, os.getcwd())
from functools import partial
import torch
import warnings 
warnings.filterwarnings("ignore")

from configs.config_promise import get_experiment_config
from horizonms.samplers import PatientSampler
from horizonms.configs import config
from horizonms.engine import utils
from horizonms.datasets import PromiseSegmentation
from horizonms.builder import build_models
from horizonms.training_utils import training


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n_exp', default=0, type=int,
                        help='the index of experiments')
    args = parser.parse_args()

    utils.init_distributed_mode(args)
    torch.backends.cudnn.benchmark = True
    torch.multiprocessing.set_sharing_strategy('file_system')
    # torch.set_num_threads(2)
    print("torch threads: ", torch.get_num_threads())

    n_exp = args.n_exp
    optimizer = 'Adam'
    scheduler = 'ReduceLROnPlateau'
    _C = get_experiment_config(optimizer, scheduler)
    ## mil baseline
    if n_exp == 0:
        cfg1 = {'model_params': {'loss_params': 
                          [dict(name='MILApproxUnaryBaselineSigmoidLoss',
                                loss_mode='ce_all', approx_method='softmax', approx_alpha=4, 
                                focal_params=dict(alpha=0.25, gamma=2.0), loss_weight=1.0),
                           dict(name='MILPairwiseLoss', softmax=False, exp_coef=-1, loss_weight=10.0)]},
                'save_params': {'experiment_name': 'residual_baseline_approx_softmax=4'}}
        cfg2 = {'model_params': {'loss_params': 
                          [dict(name='MILApproxUnaryBaselineSigmoidLoss',
                                loss_mode='ce_all', approx_method='softmax', approx_alpha=6, 
                                focal_params=dict(alpha=0.25, gamma=2.0), loss_weight=1.0),
                           dict(name='MILPairwiseLoss', softmax=False, exp_coef=-1, loss_weight=10.0)]},
                'save_params': {'experiment_name': 'residual_baseline_approx_softmax=6'}}
        cfg3 = {'model_params': {'loss_params': 
                          [dict(name='MILApproxUnaryBaselineSigmoidLoss',
                                loss_mode='ce_all', approx_method='softmax', approx_alpha=8, 
                                focal_params=dict(alpha=0.25, gamma=2.0), loss_weight=1.0),
                           dict(name='MILPairwiseLoss', softmax=False, exp_coef=-1, loss_weight=10.0)]},
                'save_params': {'experiment_name': 'residual_baseline_approx_softmax=8'}}
        _C_array = []
        for c in [cfg1, cfg2, cfg3]:
            _C_array.append(config.config_updates(_C, c))  
    ## generalized mil
    elif n_exp == 1:
        angle = (-40,41,20)
        cfg1 = {'model_params': {'loss_params': 
                          [dict(name='MILApproxUnaryParallelSigmoidLoss',
                                loss_mode='focal', angle_params=angle, approx_method='softmax',
                                approx_alpha=4, loss_weight=1),
                           dict(name='MILPairwiseLoss', softmax=False, exp_coef=-1,loss_weight=10.0)]},
               'save_params': {'experiment_name': f'residual_parallel_approx_focal_{-angle[0]}_{angle[-1]}_softmax=4'}}
        cfg2 = {'model_params': {'loss_params': 
                          [dict(name='MILApproxUnaryParallelSigmoidLoss',
                                loss_mode='focal', angle_params=angle, approx_method='quasimax',
                                approx_alpha=6, loss_weight=1),
                           dict(name='MILPairwiseLoss', softmax=False, exp_coef=-1,loss_weight=10.0)]},
               'save_params': {'experiment_name': f'residual_parallel_approx_focal_{-angle[0]}_{angle[-1]}_quasimax=6'}}
        _C_array = []
        for c in [cfg2]:
            _C_array.append(config.config_updates(_C, c))        
    ## polar transformation based mil
    elif n_exp == 2:
        osh = [90, 30]
        weight_min = 0.6
        cfg1 = {'model_params': {'loss_params': 
                          [dict(name='MILApproxUnaryPolarSigmoidLoss',
                                loss_mode='focal', weight_min=weight_min, center_mode='estimated',
                                approx_method='softmax', approx_alpha=0.5,
                                pt_params={"output_shape": osh, "scaling": "linear"},
                                focal_params={'alpha':0.25, 'gamma':2.0}, loss_weight=1.0),
                           dict(name='MILPairwiseLoss', softmax=False, exp_coef=-1, loss_weight=10.0)]},
                'save_params': {'experiment_name': f'residual_polarw_{weight_min}_approx_focal_{osh[0]}_{osh[1]}_softmax=0.5'}}
        osh = [60, 30]
        weight_min = 0.5
        cfg2 = {'model_params': {'loss_params': 
                          [dict(name='MILApproxUnaryPolarSigmoidLoss',
                                loss_mode='focal', weight_min=weight_min, center_mode='estimated',
                                approx_method='softmax', approx_alpha=0.5,
                                pt_params={"output_shape": osh, "scaling": "linear"},
                                focal_params={'alpha':0.25, 'gamma':2.0}, loss_weight=1.0),
                           dict(name='MILPairwiseLoss', softmax=False, exp_coef=-1, loss_weight=10.0)]},
                'save_params': {'experiment_name': f'residual_polarw_{weight_min}_approx_focal_{osh[0]}_{osh[1]}_quasimax=0.5'}}
        _C_array = []
        for c in [cfg1, cfg2]:
            _C_array.append(config.config_updates(_C, c)) 
    ## polar transformation assisting mil (the proposed approach)
    elif n_exp == 3:
        osh = [90, 20]
        approx_alpha = 2
        weight_min = 0.5
        cfg1 = {'model_params': {'loss_params': 
                           [dict(name='MILApproxUnaryPolarSigmoidLoss',
                                loss_mode='focal', weight_min=weight_min, center_mode='estimated',
                                approx_method='softmax', approx_alpha=approx_alpha,
                                pt_params={"output_shape": osh, "scaling": "linear"},
                                focal_params={'alpha':0.25, 'gamma':2.0}, loss_weight=1.0),
                            dict(name='MILApproxUnaryParallelSigmoidLoss',
                                loss_mode='focal', angle_params=(-40,41,20), 
                                approx_method='softmax', approx_alpha=4, loss_weight=1.0),
                            dict(name='MILPairwiseLoss', softmax=False, exp_coef=-1, loss_weight=10.0)]},
                'save_params': {'experiment_name': f'residual_parallel_polarw_{weight_min}_approx_focal_{osh[0]}_{osh[1]}_softmax={approx_alpha}'}}
        osh = [120, 20]
        approx_alpha = 2
        weight_min = 0.4
        cfg2 = {'model_params': {'loss_params': 
                           [dict(name='MILApproxUnaryPolarSigmoidLoss',
                                loss_mode='focal', weight_min=weight_min, center_mode='estimated',
                                approx_method='softmax', approx_alpha=approx_alpha,
                                pt_params={"output_shape": osh, "scaling": "linear"},
                                focal_params={'alpha':0.25, 'gamma':2.0}, loss_weight=1.0),
                            dict(name='MILApproxUnaryParallelSigmoidLoss',
                                loss_mode='focal', angle_params=(-40,41,20), 
                                approx_method='softmax', approx_alpha=4, loss_weight=1.0),
                            dict(name='MILPairwiseLoss', softmax=False, exp_coef=-1, loss_weight=10.0)]},
                'save_params': {'experiment_name': f'residual_parallel_polarw_{weight_min}_approx_focal_{osh[0]}_{osh[1]}_quasimax={approx_alpha}'}}
        _C_array = []
        for c in [cfg1, cfg2]:
            _C_array.append(config.config_updates(_C, c)) 
    ## pos = bbox, neg = generalized mil
    elif n_exp == 4:
        cfg1 = {'model_params': {'loss_params': 
                           [dict(name='MILApproxUnaryBboxPosGeneralizedNegSigmoidLoss',
                                loss_mode='focal', approx_method='softmax', approx_alpha=0.5,
                                focal_params={'alpha':0.25, 'gamma':2.0}, loss_weight=1.0),
                            dict(name='MILPairwiseLoss', softmax=False, exp_coef=-1, loss_weight=10.0)]},
                'save_params': {'experiment_name': f'residual_bboxpos_approx_focal_softmax=0.5'}}
        cfg2 = {'model_params': {'loss_params': 
                           [dict(name='MILApproxUnaryBboxPosGeneralizedNegSigmoidLoss',
                                loss_mode='focal', approx_method='softmax', approx_alpha=1,
                                focal_params={'alpha':0.25, 'gamma':2.0}, loss_weight=1.0),
                            dict(name='MILPairwiseLoss', softmax=False, exp_coef=-1, loss_weight=10.0)]},
                'save_params': {'experiment_name': f'residual_bboxpos_approx_focal_softmax=1'}}
        cfg3 = {'model_params': {'loss_params': 
                           [dict(name='MILApproxUnaryBboxPosGeneralizedNegSigmoidLoss',
                                loss_mode='focal', approx_method='softmax', approx_alpha=2,
                                focal_params={'alpha':0.25, 'gamma':2.0}, loss_weight=1.0),
                            dict(name='MILPairwiseLoss', softmax=False, exp_coef=-1, loss_weight=10.0)]},
                'save_params': {'experiment_name': f'residual_bboxpos_approx_focal_softmax=2'}}
        cfg4 = {'model_params': {'loss_params': 
                           [dict(name='MILApproxUnaryBboxPosGeneralizedNegSigmoidLoss',
                                loss_mode='focal', approx_method='softmax', approx_alpha=4,
                                focal_params={'alpha':0.25, 'gamma':2.0}, loss_weight=1.0),
                            dict(name='MILPairwiseLoss', softmax=False, exp_coef=-1, loss_weight=10.0)]},
                'save_params': {'experiment_name': f'residual_bboxpos_approx_focal_softmax=4'}}
        _C_array = []
        for c in [cfg1, cfg2, cfg3, cfg4]:
            _C_array.append(config.config_updates(_C, c))  
    
    for _C in _C_array:
        random = False
        margin = 0
        base_experiment_name = _C['save_params']['experiment_name']
        if margin == 0:
            cfg = {}
        else:
            if random:
                cfg = {'save_params': {'experiment_name': base_experiment_name + \
                            '_margin='+str(margin)+'_random'}}
            else:
                cfg = {'save_params': {'experiment_name': base_experiment_name + \
                            '_margin='+str(margin)}}
        _C_used = config.config_updates(_C, cfg)
        assert _C_used['save_params']['experiment_name'] is not None, \
                    "experiment_name has to be set"

        train_params = _C_used['train_params']
        data_params = _C_used['data_params']
        model_params = _C_used['model_params']
        dataset_params = _C_used['dataset_params']
        save_params = _C_used['save_params']

        output_dir = os.path.join(save_params['dir_save'],save_params['experiment_name'])
        os.makedirs(output_dir, exist_ok=True)
        config.save_config_file(os.path.join(output_dir, 'config.yaml'), _C_used)
        print("saving files to {:s}".format(output_dir))
        device = torch.device(_C_used['device'])

        dataset_params_common = dict(
            root=dataset_params['root_path'], 
            margin=margin, random=random,
            transforms_pt=data_params['transforms'],
            to_tensor=True)
        dataset_params_train = dict(image_folder=dataset_params['train_path'][0], 
                                    gt_folder=dataset_params['train_path'][1])
        dataset_params_valid = dict(image_folder=dataset_params['valid_path'][0], 
                                    gt_folder=dataset_params['valid_path'][1])
        sibright_dataset = partial(PromiseSegmentation, **dataset_params_common)
        dataset = sibright_dataset(**dataset_params_train)
        if dataset_params_valid is not None:
            dataset_test = sibright_dataset(**dataset_params_valid)
            print(f"#train = {len(dataset)}, #test = {len(dataset_test)}")
        else:
            print(f"#train = {len(dataset)}, no testing data is provided.")

        print("Creating data loaders")
        train_sampler = torch.utils.data.RandomSampler(dataset)
        train_batch_sampler = torch.utils.data.BatchSampler(
                train_sampler, train_params['batch_size'], drop_last=True)
        test_patient_sampler = PatientSampler(dataset_test, dataset_params['grp_regex'], shuffle=False)

        train_loader = torch.utils.data.DataLoader(
            dataset, batch_sampler=train_batch_sampler, num_workers=data_params['workers'],
            collate_fn=utils.collate_fn, pin_memory=True)

        valid_loader = torch.utils.data.DataLoader(
            dataset_test, batch_size=1,
            batch_sampler=test_patient_sampler, num_workers=data_params['workers'],
            collate_fn=utils.collate_fn, pin_memory=True)

        print("Creating model with parameters: {}".format(model_params))
        model = build_models(model_params)
        model.to(device)

        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.rank],
                find_unused_parameters=False)

        training(train_loader, valid_loader, model, train_params, 
                scheduler=scheduler, train_sampler=train_sampler,
                distributed=args.distributed, output_dir=output_dir)