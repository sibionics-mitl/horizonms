import os, sys
sys.path.insert(0, os.getcwd())
import torch
import warnings
import copy
warnings.filterwarnings("ignore")

from configs.config_voc import get_experiment_config
from horizonms.configs import config
from horizonms.datasets import VOCDetection
from horizonms.engine import utils
from horizonms.training_utils import create_dataloader, training
from horizonms.builder import build_models


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n_exp', default=1, type=int,
                        help='the index of experiments')
    # distributed training parameters
    parser.add_argument('--world-size', default=0, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    args = parser.parse_args()

    utils.init_distributed_mode(args)
    torch.backends.cudnn.benchmark = True
    torch.multiprocessing.set_sharing_strategy('file_system')
    # torch.set_num_threads(2)
    # print("torch threads: ", torch.get_num_threads())

    n_exp = args.n_exp
    if n_exp == 1:
        net_name = 'YOLOv1'
        scheduler = 'MultiStepLR'
        _C = get_experiment_config(net_name, scheduler=scheduler)
        cfg = {'save_params': {'experiment_name': 'yolov1'}}
        _C_base = config.config_updates(_C, cfg)    
    
    print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    print(f"training {_C_base['model_params']['net_params']['name']}")
    print()
    _C_used = copy.deepcopy(_C_base)
    assert _C_used['save_params']['experiment_name'] is not None, "experiment_name has to be set"

    train_params = _C_used['train_params']
    data_params = _C_used['data_params']
    model_params = _C_used['model_params']
    dataset_params = _C_used['dataset_params']
    save_params = _C_used['save_params']

    output_dir = os.path.join(save_params['dir_save'], save_params['experiment_name'])
    os.makedirs(output_dir, exist_ok=True)
    config.save_config_file(os.path.join(output_dir, 'config.yaml'), _C_used)
    print("saving files to {:s}".format(output_dir))

    device = torch.device(_C_used['device'])
    epoch_per_save = train_params['epoch_per_save']
    print(f"training in device = {device}")    
    
    sample_weighting = data_params.get('sample_weighting', None)
    dataset_params_common = dict(
        root=dataset_params['root_path'], 
        keep_difficult=dataset_params['keep_difficult'],
        to_tensor=True)
    dataset_params_train = dict(years=dataset_params['train_years'],
                                image_sets=dataset_params['train_sets'],
                                transforms_cv=data_params['transforms_train'])
    dataset_params_valid = dict(years=dataset_params['test_years'], 
                                image_sets=dataset_params['test_sets'],
                                transforms_cv=data_params['transforms_test'])
    sample_weighting = data_params.get('sample_weighting', None)
    train_loader, valid_loader, train_sampler = create_dataloader(
        VOCDetection, dataset_params_common, 
        dataset_params_train, dataset_params_valid=dataset_params_valid,
        sample_weighting=sample_weighting,
        batch_size=train_params['batch_size'], 
        workers=_C_used['workers'], distributed=args.distributed)
   
    print(f"Creating model with parameters: {model_params}")   
    model = build_models(model_params)
    model.to(device)    

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.rank],
            find_unused_parameters=False)

    training(train_loader, valid_loader, model, train_params, 
             scheduler=scheduler, train_sampler=train_sampler,
             distributed=args.distributed, output_dir=output_dir)