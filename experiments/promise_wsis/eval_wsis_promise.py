import os, sys
sys.path.insert(0, os.getcwd())
import numpy as np
import pandas as pd
import torch
import warnings 
warnings.filterwarnings("ignore")

from horizonms.samplers import PatientSampler
from horizonms.configs import config
from horizonms.utils import pd_utils 
from horizonms.engine import utils
from horizonms.datasets import PromiseSegmentation
from horizonms.builder import build_models


@torch.no_grad()
def evaluate(epoch, model, data_loader, image_names, device, threshold, save_detection=None, smooth=1e-10):
    file_2d = os.path.join(save_detection,'dice_2d.xlsx')
    file_3d = os.path.join(save_detection,'dice_3d.xlsx')
    torch.set_num_threads(1)
    model.eval()

    nn = 0
    dice_2d, dice_3d = {k:[] for k in range(len(threshold))}, {k:[] for k in range(len(threshold))}
    for images, targets in data_loader:
        nn = nn + 1
        # print("{}/{}".format(nn,len(data_loader))) 

        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        gt = torch.stack([t["masks"].value for t in targets], dim=0)
        gt = gt.bool()
        outputs = model.predict_one_batch(images)
                
        for n_th,th in enumerate(threshold):
            pred = outputs>th
            intersect = pred&gt
            v_dice_2d = (2*torch.sum(intersect,dim=(1,2,3))+smooth)/(torch.sum(pred,dim=(1,2,3))+torch.sum(gt,dim=(1,2,3))+smooth)
            v_dice_3d = (2*torch.sum(intersect)+smooth)/(torch.sum(pred)+torch.sum(gt)+smooth)
            dice_2d[n_th].append(v_dice_2d.cpu().numpy())
            dice_3d[n_th].append(v_dice_3d.cpu().numpy())

    dice_2d = [np.hstack(dice_2d[key]) for key in dice_2d.keys()]
    dice_3d = [np.hstack(dice_3d[key]) for key in dice_3d.keys()]
    dice_2d = np.vstack(dice_2d).T
    dice_3d = np.vstack(dice_3d).T
    
    dice_2d = pd.DataFrame(data=dice_2d, columns=threshold)
    dice_3d = pd.DataFrame(data=dice_3d, columns=threshold)
    
    pd_utils.append_df_to_excel(file_2d, dice_2d, sheet_name=str(epoch), index=False)
    pd_utils.append_df_to_excel(file_3d, dice_3d, sheet_name=str(epoch), index=False)

    mean_2d = np.mean(dice_2d, axis=0)
    std_2d = np.std(dice_2d, axis=0)
    loc2 = np.argmax(mean_2d)
    mean_3d = np.mean(dice_3d, axis=0)
    std_3d = np.std(dice_3d, axis=0)
    loc3 = np.argmax(mean_3d)
    print('2d mean: {}({})'.format(mean_2d.iloc[loc2],std_2d.iloc[loc2]))
    print('3d mean: {}({})'.format(mean_3d.iloc[loc3],std_3d.iloc[loc3]))


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n_exp', default=1, type=int,
                        help='the index of experiments')
    args = parser.parse_args()

    utils.init_distributed_mode(args)
    print(args)

    n_exp = args.n_exp
    dir_save_root = os.path.join('results','promise')
    threshold = [0.001,0.005,0.01]+list(np.arange(0.05,0.9,0.05))
    ## mil baseline
    if n_exp==0:
        experiment_names = [
            'residual_baseline_approx_softmax=4',
            'residual_baseline_approx_softmax=6',
            'residual_baseline_approx_softmax=8',
        ]
    ## generalized mil
    elif n_exp==1:
        experiment_names = [
            'residual_parallel_approx_focal_40_20_softmax=4',
            'residual_parallel_approx_focal_40_20_quasimax=6',
        ] 
    ## polar transformation based mil
    elif n_exp==2:
        experiment_names = [
            'residual_polarw_0.6_approx_focal_90_30_softmax=0.5',
            'residual_polarw_0.5_approx_focal_60_30_quasimax=0.5',
        ]
    ## polar transformation assisting mil (the proposed approach)
    elif n_exp==3:
        experiment_names = [
            # 'residual_parallel_polarw_approx_focal_120_10_softmax=1',
            # 'residual_parallel_polarw_approx_focal_120_10_quasimax=0.5',
            'residual_parallel_polarw_0.5_approx_focal_90_20_softmax=2',
            'residual_parallel_polarw_0.4_approx_focal_120_20_quasimax=2',
        ]
    ## pos = bbox, neg = generalized mil
    elif n_exp==4:
        experiment_names = [
            'residual_bboxpos_approx_focal_softmax=0.5',
            'residual_bboxpos_approx_focal_softmax=1',
            'residual_bboxpos_approx_focal_softmax=2',
            'residual_bboxpos_approx_focal_softmax=4',
        ]
     
    for base_experiment_name in experiment_names:
        random = False
        margin = 0
        if margin == 0:
            experiment_name = base_experiment_name
        else:
            if random:
                experiment_name = f"{base_experiment_name}_margin={margin}_random"
            else:
                experiment_name = f"{base_experiment_name}_margin={margin}"
        print(experiment_name)
        output_dir = os.path.join(dir_save_root, experiment_name)
        _C = config.read_config_file(os.path.join(output_dir, 'config.yaml'))
        assert _C['save_params']['experiment_name']==experiment_name, "experiment_name is not right"
        cfg = {'data_params': {'workers': 4}}
        _C = config.config_updates(_C, cfg)

        train_params = _C['train_params']
        data_params = _C['data_params']
        model_params = _C['model_params']
        dataset_params = _C['dataset']
        save_params = _C['save_params']

        device = torch.device(_C['device'])      

        # Data loading code
        print("Loading data")
        dataset_test = PromiseSegmentation(
                    root=dataset_params['root_path'], 
                    image_folder=dataset_params['valid_path'][0], 
                    gt_folder=dataset_params['valid_path'][1], 
                    margin=margin, random=random,
                    transforms_pt=data_params['transforms'])
        image_names = dataset_test.image_names
        
        print("Creating data loaders")
        test_patient_sampler = PatientSampler(dataset_test, dataset_params['grp_regex'], shuffle=False)

        data_loader_test = torch.utils.data.DataLoader(
                dataset_test, batch_size=1,
                batch_sampler=test_patient_sampler, num_workers=data_params['workers'],
                collate_fn=utils.collate_fn, pin_memory=True)

        print("Creating model with parameters: {}".format(model_params))
        model = build_models(model_params)
        model.to(device)
        
        file_2d = os.path.join(output_dir,'dice_2d.xlsx')
        file_3d = os.path.join(output_dir,'dice_3d.xlsx')
        # if os.path.exists(file_2d):
        #     os.remove(file_2d)
        # if os.path.exists(file_3d):
        #     os.remove(file_3d)
        for epoch in range(50):
            model_file = 'model_{:02d}'.format(epoch)
            print('loading model {}.pth'.format(model_file))
            checkpoint = torch.load(os.path.join(output_dir, model_file+'.pth'), map_location='cpu')
            model.load_state_dict(checkpoint['model'], strict=False)
        
            print('start evaluating {} ...'.format(epoch))
            model.training = False
            evaluate(epoch, model, data_loader_test, image_names=image_names, device=device, threshold=threshold, save_detection=output_dir)
            
        dice_2d_all = pd.read_excel(file_2d, sheet_name=None)
        dice_3d_all = pd.read_excel(file_3d, sheet_name=None)