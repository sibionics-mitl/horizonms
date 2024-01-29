import os, sys
sys.path.insert(0, os.getcwd())
import numpy as np
import pandas as pd
import cv2
import warnings
import tqdm
from collections import defaultdict
import torch
warnings.filterwarnings("ignore")

from horizonms.configs import config
from horizonms.datasets import VOCDetection
from horizonms.engine import utils
from horizonms.testing_utils import create_testing_dataloader
from horizonms.builder import build_models


Color = [[0, 0, 0],
        [128, 0, 0],
        [0, 128, 0],
        [128, 128, 0],
        [0, 0, 128],
        [128, 0, 128],
        [0, 128, 128],
        [128, 128, 128],
        [64, 0, 0],
        [192, 0, 0],
        [64, 128, 0],
        [192, 128, 0],
        [64, 0, 128],
        [192, 0, 128],
        [64, 128, 128],
        [192, 128, 128],
        [0, 64, 0],
        [128, 64, 0],
        [0, 192, 0],
        [128, 192, 0],
        [0, 64, 128]]


def prediction_visualization(image, voc_classes, predictions):
    for box, label, score in zip(predictions['boxes'], predictions['labels'], predictions['scores']):
        color = Color[label]
        box = np.around(box.to('cpu').numpy()).astype(int)
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, 2)
        text = f"{voc_classes[label]}: {round(score.item(), 2)}"
        text_size, baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        p1 = (box[0], box[1]- text_size[1])
        cv2.rectangle(image, (p1[0] - 2//2, p1[1] - 2 - baseline), (p1[0] + text_size[0], p1[1] + text_size[1]), color, -1)
        cv2.putText(image, text, (p1[0], p1[1] + baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, 8)
    return image


def gt_visualization(image, voc_classes, gts):
    for box in gts:
        label = int(box[-1])
        box = np.around(box[:4]).astype(int)
        color = Color[label]
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, 2)
        text = f"gt: {voc_classes[label]}"
        text_size, baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        p1 = (box[0], box[1]- text_size[1])
        cv2.rectangle(image, (p1[0] - 2//2, p1[1] - 2 - baseline), (p1[0] + text_size[0], p1[1] + text_size[1]), color, -1)
        cv2.putText(image, text, (p1[0], p1[1] + baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, 8)
    return image


def voc_ap(rec, prec, use_07_metric=False):
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0.,1.1,0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec>=t])
            ap = ap + p/11.
    else:
        # correct ap caculation
        mrec = np.concatenate(([0.],rec,[1.]))
        mpre = np.concatenate(([0.],prec,[0.]))
        for i in range(mpre.size -1, 0, -1):
            mpre[i-1] = np.maximum(mpre[i-1],mpre[i])

        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(preds, target, voc_classes, threshold=0.5, use_07_metric=False):
    '''
    preds {'cat':[[image_id,confidence,x1,y1,x2,y2],...],'dog':[[],...]}
    target {(image_id,class):[[],]}
    '''
    aps = []
    msgs = []
    for i, class_ in enumerate(voc_classes):
        pred = preds[class_] #[[image_id,confidence,x1,y1,x2,y2],...]
        if len(pred) == 0: #如果这个类别一个都没有检测到的异常情况
            ap = -1
            msg = f"class {class_}: ap = {ap:0.4f}"
            msgs.append(msg)
            print(msg)
            aps += [ap]
            break
        #print(pred)
        image_ids = [x[0] for x in pred]
        confidence = np.array([float(x[1]) for x in pred])
        BB = np.array([x[2:] for x in pred])
        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

        # go down dets and mark TPs and FPs
        npos = 0.
        for (key1,key2) in target:
            if key2 == class_:
                npos += len(target[(key1,key2)]) #统计这个类别的正样本，在这里统计才不会遗漏
        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        for d,image_id in enumerate(image_ids):
            bb = BB[d] #预测框
            if (image_id,class_) in target:
                BBGT = target[(image_id,class_)] #[[],]
                for bbgt in BBGT:
                    # compute overlaps
                    # intersection
                    ixmin = np.maximum(bbgt[0], bb[0])
                    iymin = np.maximum(bbgt[1], bb[1])
                    ixmax = np.minimum(bbgt[2], bb[2])
                    iymax = np.minimum(bbgt[3], bb[3])
                    iw = np.maximum(ixmax - ixmin + 1., 0.)
                    ih = np.maximum(iymax - iymin + 1., 0.)
                    inters = iw * ih

                    union = (bb[2]-bb[0]+1.)*(bb[3]-bb[1]+1.) + (bbgt[2]-bbgt[0]+1.)*(bbgt[3]-bbgt[1]+1.) - inters
                    if union == 0:
                        print(bb,bbgt)
                    
                    overlaps = inters/union
                    if overlaps > threshold:
                        tp[d] = 1
                        BBGT.remove(bbgt) #这个框已经匹配到了，不能再匹配
                        if len(BBGT) == 0:
                            del target[(image_id,class_)] #删除没有box的键值
                        break
                fp[d] = 1-tp[d]
            else:
                fp[d] = 1
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp/float(npos)
        prec = tp/np.maximum(tp + fp, np.finfo(np.float64).eps)
        #print(rec,prec)
        ap = voc_ap(rec, prec, use_07_metric)
        msg = f"class {class_}: ap = {ap:0.4f}"
        msgs.append(msg)
        print(msg)
        aps += [ap]
    msg = f"map = {np.mean(aps):0.4f}"
    msgs.append(msg)
    print(msg)
    return msgs


@torch.no_grad()
def detection_evaluation(model, data_loader_test, device, voc_classes, save_dir=None):
    model.eval()
    ytrues, ypreds = defaultdict(list), defaultdict(list)
    for images, targets in tqdm.tqdm(data_loader_test):
        images = list(image.to(device) for image in images)
        targets = [{k: v for k, v in t.items()} for t in targets]
        ytrue_batch = [t["bboxes"].value.numpy() for t in targets]
        ypred_batch = model.predict_one_batch(images)
        image_id_batch = [int(t['image_id'].value.item()) for t in targets]

        for image_id, ytrue, ypred in zip(image_id_batch, ytrue_batch, ypred_batch):
            for gt in ytrue:
                label = int(gt[-1])
                ytrues[(image_id, voc_classes[label])].append(list(gt[:4]))

            for box, label, score in zip(ypred["boxes"], ypred["labels"], ypred["scores"]):
                class_name = voc_classes[label.item()]
                ypreds[class_name].append([image_id, score.item()]+list(box.cpu().numpy()))
            
        if save_dir is not None:
            # plot prediction
            for image, gt, ypred in zip(images, targets, ypred_batch):
                image = 255 * (image - image.min()) / (image.max() - image.min())
                image = image.permute(1,2,0).to('cpu').numpy().astype(np.uint8)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                image_id = int(gt['image_id'].value.item())
                gt_boxes = gt['bboxes'].value.to('cpu').numpy()

                image_gt = np.copy(image)
                image_gt = gt_visualization(image_gt, voc_classes, gt_boxes)
                cv2.imwrite(os.path.join(save_dir, f"{image_id}_gt.jpg"), image_gt)

                image_pred = np.copy(image)
                image_pred = prediction_visualization(image_pred, voc_classes, ypred)
                cv2.imwrite(os.path.join(save_dir, f"{image_id}_pd.jpg"), image_pred)
        
    return ytrues, ypreds


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n_exp', default=1, type=int,
                        help='the index of experiments')
    parser.add_argument('--world-size', default=0, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://',
                        help='url used to set up distributed training')
    args = parser.parse_args()

    utils.init_distributed_mode(args)

    torch.backends.cudnn.benchmark = True
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.set_num_threads(2)

    print(args)
    n_exp = args.n_exp
    dir_save_root = os.path.join('results', 'voc')
    if n_exp == 1:
        experiment_name = 'yolov1'
    
    output_dir = os.path.join(dir_save_root, experiment_name)
    print(output_dir)
    _C = config.read_config_file(os.path.join(output_dir, 'config.yaml'))
    assert _C['save_params']['experiment_name']==experiment_name, "experiment_name is not right"
    cfg = {'workers': 32}
    _C = config.config_updates(_C, cfg)

    train_params = _C['train_params']
    data_params = _C['data_params']
    model_params = _C['model_params']
    dataset_params = _C['dataset_params']
    save_params = _C['save_params']

    device = torch.device(_C['device'])
    print(f"training in device = {device}")

    print("Creating model with parameters: {}".format(model_params))
    model = build_models(model_params)
    model.to(device).eval()    

    epoches = list(range(train_params['epoch_save_start'], train_params['epochs']))
    summary = pd.read_csv(os.path.join(output_dir, "summary.csv"))
    loc = summary['val_loss'].argmin()
    best_epoch = summary['epoch'].iloc[loc]
    print(f"best epoches = {best_epoch}")
        
    model_file = 'model_{:02d}'.format(best_epoch)
    print('loading model {}.pth'.format(model_file))
    checkpoint = torch.load(os.path.join(output_dir, model_file+'.pth'), map_location='cpu')
    model.load_state_dict(checkpoint['model'])

    print('start evaluating {} ...'.format(best_epoch))   
    dataset_params = dict(
        root=dataset_params['root_path'], 
        years=dataset_params['test_years'], 
        image_sets=dataset_params['test_sets'],
        keep_difficult=dataset_params['keep_difficult'],
        transforms_cv=data_params['transforms_test'],
        to_tensor=True
    )
    dataset_test, data_loader_test = create_testing_dataloader(
                VOCDetection, dataset_params, test_batch_size=1,
                workers=_C['workers'], distributed=args.distributed)

    dir_save = None
    if dir_save is not None:
        os.makedirs(dir_save, exist_ok=True)
    ytrues, ypreds = detection_evaluation(model, data_loader_test, device, dataset_test.voc_classes, dir_save)
    results = voc_eval(ypreds, ytrues, dataset_test.voc_classes)
    file_name = os.path.join(output_dir, "results_summary.txt")
    with open(file_name, 'w') as fp:
        for item in results:
            fp.write(f"{item}\n")