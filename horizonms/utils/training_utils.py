import numpy as np

def save_checkpoints_update(save_checkpoints, val_metric_logger):
    checkpoint_keys = []
    for key, value_dict in save_checkpoints.items():
        val_value = val_metric_logger.meters[key].global_avg
        if value_dict['mode'] == 'min':
            flag = value_dict['value'] > val_value
        elif value_dict['mode'] == 'max':
            flag = value_dict['value'] < val_value
        if flag:
            checkpoint_keys.append(key)
            value_dict['value'] = val_value
    return checkpoint_keys, save_checkpoints

def summary_update(epoch, nb_save, summary={'epoch': []},
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