import math
import os, sys
import pandas as pd
import numpy as np
import time
import datetime
import torch
import matplotlib
matplotlib.use('agg')
import matplotlib.pylab as plt

from ..datasets import TrainDataLoaderIter, ValidDataLoaderIter
from .utils import MetricLogger, SmoothedValue, warmup_lr_scheduler
from .utils import reduce_dict, save_on_master
from .lr_finder import LRFinder
from .early_stop import EarlyStopping
from .base import CheckpointMetric, save_checkpoints_update


class Trainer():
    r"""model traning class.

    Args:
        model (nn.Module): the network.
        optimizer (callable): optimizer for network optimization.
        train_loader (callable): data loader for training subset.
        valid_loader (callable): data loader for validation subset.
        lr_scheduler (float): lr scheduler.
        lr_scheduler_mode (str): lr scheduler is updated in each `'epoch'` or in each `'iteration'`.
        lr_scheduler_metric (bool): if True, update lr scheduler based on validation loss.
        warmup_epochs (int): number of epochs for warmup.
        distributed (bool): if True, distributed training is conducted.
        train_sampler (callable): sampler for training subset.
        epochs (int): total epochs for training.
        non_blocking (bool): if True, achieves non_blocking transfer to GPU.
        early_stop_patience (int): number of epcohs for early stop.
        output_dir (str): directory for model output.
    """
    def __init__(self, model, optimizer, train_loader, valid_loader=None, lr_scheduler=None,
                 lr_scheduler_mode='epoch', lr_scheduler_metric=True, warmup_epochs=0,
                 distributed=True, train_sampler=None, epochs=80, non_blocking=True,
                 early_stop_patience=None, output_dir='./'):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.lr_scheduler = lr_scheduler
        assert lr_scheduler_mode in ['epoch', 'iteration'], \
            f"lr_scheduler_mode has to be 'epoch' or 'iteration', but got {lr_scheduler_mode}"
        self.lr_scheduler_mode = lr_scheduler_mode
        self.lr_scheduler_metric = lr_scheduler_metric
        self.warmup_epochs = warmup_epochs
        self.distributed = distributed
        self.train_sampler = train_sampler
        self.epochs = epochs
        self.non_blocking = non_blocking
        if (early_stop_patience is None) | (early_stop_patience == 0):
            self.early_stop = None
        else:
            self.early_stop = EarlyStopping(patience=early_stop_patience)
        self.output_dir = output_dir

        if "cuda" in str(next(self.model.parameters()).device):
            self.device = "cuda"
        else:
            self.device = "cpu"

        if distributed:
            self.model_without_ddp = model.module
        else:
            self.model_without_ddp = self.model

        self.summary = {'epoch': []}

    def set_lr_scheduler(self, lr_scheduler, lr_scheduler_metric):
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_metric = lr_scheduler_metric

    def get_warmup_lr_scheduler(self, warmup_epochs):
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, warmup_epochs * len(self.train_loader) - 1)
        self.lr_scheduler_warmup = warmup_lr_scheduler(self.optimizer, warmup_iters, warmup_factor)

    def lr_finder(self, start_lr=None, end_lr=1.0, num_iter=100, step_mode='linear',
                  smooth_f=0.05, diverge_th=5, accumulation_steps=1):
        train_loader = TrainDataLoaderIter(self.train_loader)
        val_loader = ValidDataLoaderIter(self.valid_loader)
        lr_finder = LRFinder(self.model, self.optimizer, cache_dir=self.output_dir)
        lr_finder.range_test(train_loader, val_loader, start_lr, end_lr, num_iter, step_mode,
        smooth_f, diverge_th, accumulation_steps, self.non_blocking)
        outputs = lr_finder.plot(skip_start=0, skip_end=0)
        if len(outputs) == 1:
            print("No max_lr is found.")
            sys.exit(1)
        else:
            ax, max_lr = outputs
        return max_lr

    def train(self, sub_epoch=1, epoch_per_save=1, epoch_save_start=0,
            save_checkpoints=[CheckpointMetric(name='save_all', mode='all')],
            clipnorm=0.001, print_freq=20):
        r"""model training. An 'equivalent epoch' is defined during training, which is 
        used as the unit for the number of training iterations. 'Equivalent epoch' is 
        especially useful when the dataset is small.

        Args:
            sub_epoch (int): equivalent epoch. Default: 1.
            epoch_per_save (int): save model in each `epoch_per_save` equivalent epochs. Default: 1.
            epoch_save_start (int): save model after `epoch_save_start` equivalent epochs. Default: 0.
            save_checkpoints (List[CheckpointMetric]): determine which checkpoints are saved.
                Default: `[CheckpointMetric(name='save_all', mode='all')]`.
            clipnorm (float): clip norm. Default: 0.001.
            print_freq (int): print training information in each `print_freq` iterations.
        """
        self.get_warmup_lr_scheduler(sub_epoch * self.warmup_epochs)
        print("Start training")
        start_time = time.time()
        for epoch in range(self.epochs):
            # Set epoch count for DistributedSampler.
            # We don't need to set_epoch for the validation sampler as we don't want
            # to shuffle for validation.
            if self.distributed:
                self.train_sampler.set_epoch(epoch)
        
            # do training
            for _ in range(sub_epoch):
                metric_logger = self.train_one_epoch(epoch, clipnorm, print_freq)

            # do evaluation
            if (self.valid_loader is not None) & ((epoch+1) % epoch_per_save == 0):
                val_metric_logger = self.evaluate_one_epoch()
                checkpoint_keys, save_checkpoints = save_checkpoints_update(save_checkpoints,
                                                                            val_metric_logger)
                val_loss = val_metric_logger.meters['val_loss'].global_avg
                if (epoch == self.epochs-1) | ((epoch >= epoch_save_start)
                        & (len(checkpoint_keys) > 0)):
                    lr_scheduler_dict = self.lr_scheduler.state_dict()
                    if '_scale_fn_ref' in lr_scheduler_dict.keys():
                        lr_scheduler_dict.pop('_scale_fn_ref')
                    save_on_master({
                        'model': self.model_without_ddp.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler_dict,
                        'epoch': epoch},
                        os.path.join(self.output_dir, f'model_{epoch:02}.pth'))

                nb_save = (epoch+1) // epoch_per_save
                # collect the results and save
                self.summary = _summary_update(epoch, nb_save, self.summary, 
                                                  metric_logger, val_metric_logger)
                summary_save = pd.DataFrame(self.summary)
                summary_save.to_csv(os.path.join(self.output_dir, 'summary.csv'), index=False)

                if (self.lr_scheduler_mode == 'epoch') & (epoch >= self.warmup_epochs):
                    if self.lr_scheduler_metric:
                        self.lr_scheduler.step(val_loss)
                    else:
                        self.lr_scheduler.step()
                
                # early stop check
                if (self.early_stop is not None) & (epoch >= epoch_save_start):
                    if self.early_stop.step(val_loss):
                        print('Early stop at epoch = {}'.format(epoch))
                        break

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))

        ## plot training and validation loss
        plt.figure()
        plt.plot(summary_save['epoch'], summary_save['loss'], '-ro', label='train')
        plt.plot(summary_save['epoch'], summary_save['val_loss'], '-g+', label='valid')
        plt.legend(loc=0)
        plt.savefig(os.path.join(self.output_dir, 'loss.jpg'))
        time.sleep(2)
        print("Training done")

    def train_one_epoch(self, epoch, clipnorm=0.001, print_freq=50):
        r"""model training, one epoch of iterations are achieved. 

        Args:
            epoch (int): index of epoch.
            clipnorm (float): clip norm. Default: 0.001.
            print_freq (int): print training information in each `print_freq` iterations.
        """
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        self.model.train()
        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
        header = 'Epoch: [{}]'.format(epoch)

        for images, targets in metric_logger.log_every(self.train_loader, print_freq, header):
            images = list(image.to(self.device, non_blocking=self.non_blocking) for image in images)
            targets = [{k: v.to(self.device, non_blocking=self.non_blocking) for k, v in t.items() 
                        if not isinstance(v.value, str)} for t in targets]

            loss_dict, _ = self.model_without_ddp.forward_train(images, targets)

            losses = sum(loss for loss in loss_dict.values())

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = reduce_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            loss_value = losses_reduced.item()
            # if loss_value > 8:
            #     print(loss_value)

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                print(loss_dict_reduced)
                sys.exit(1)

            self.optimizer.zero_grad()
            losses.backward()
            if clipnorm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clipnorm)
            self.optimizer.step()

            if epoch < self.warmup_epochs:
                # warmup lr scheduler
                self.lr_scheduler_warmup.step()
            else:
                # update lr scheduler
                if self.lr_scheduler_mode == 'iteration':
                    self.lr_scheduler.step()

            metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
            metric_logger.update(lr=self.optimizer.param_groups[0]["lr"])

        return metric_logger

    @torch.no_grad()
    def evaluate_one_epoch(self):
        r"""model evaluation. 
        """
        n_threads = torch.get_num_threads()
        metric_logger = MetricLogger(delimiter="  ")
        header = 'Validation: '

        torch.set_num_threads(1)
        self.model.eval()
        loss_summary = []
        for images, targets in metric_logger.log_every(self.valid_loader, print_freq=10e5, header=header, training=False):
            images = list(image.to(self.device, non_blocking=self.non_blocking) for image in images)
            targets = [{k: v.to(self.device, non_blocking=self.non_blocking) for k, v in t.items() 
                        if not isinstance(v.value, str)} for t in targets]

            loss_dict, metric_dict, _ = self.model_without_ddp.test_one_batch(images, targets)

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = reduce_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            loss_dict_reduced = dict(('val_'+k, v) for k, v in loss_dict_reduced.items())
            loss_reduced = dict((k, v.item()) for k, v in loss_dict_reduced.items())

            if metric_dict:
                metric_dict_reduced = reduce_dict(metric_dict)
                metric_dict_reduced = dict(('val_'+k, v) for k, v in metric_dict_reduced.items())
                metric_reduced = dict((k, v.item()) for k, v in metric_dict_reduced.items())
                metric_logger.update(val_loss=losses_reduced, **loss_dict_reduced, **metric_dict_reduced)
                loss_summary.append(loss_reduced.update(metric_reduced))
            else:
                metric_logger.update(val_loss=losses_reduced, **loss_dict_reduced)
                loss_summary.append(loss_reduced)
        torch.set_num_threads(n_threads)

        return metric_logger


def _summary_update(epoch, nb_save, summary={'epoch': []},
                   metric_logger=None, val_metric_logger=None):
    summary['epoch'].append(epoch)
    for name, meter in metric_logger.meters.items():
        if name=='lr':
            v = meter.global_avg
        else:
            v = float(np.around(meter.global_avg,8))
        if name not in summary.keys():
            if nb_save != 1:
                summary[name] = [0]*(nb_save-1) + [v]  
            else:  
                summary[name] = [v]
        else:
            summary[name].append(v)
    for name, meter in val_metric_logger.meters.items():
        v = float(np.around(meter.global_avg,8))
        if name not in summary.keys():
            if nb_save != 1:
                summary[name] = [0]*(nb_save-1) + [v]  
            else:
                summary[name] = [v]
        else:
            summary[name].append(v)
    return summary