__all__ = ["CheckpointMetric", "save_checkpoints_update"]


class CheckpointMetric():
    r"""Update metric for checkpoint. Three modes can be set for each metric, that is, `'min'`, `'max'`, and `'all'`.

    Args:
        name (str): name of the metric.
        mode (str): mode of the metric. Its value is `'min'`, `'max'`, or `'all'`.
    """
    def __init__(self, name, mode):
        assert mode in ['min', 'max', 'all'], "mode has to be in 'min', 'max' or 'all'"
        self.name = name
        self.mode = mode
        if mode == 'min':
            self.value = 10000
        elif mode == 'max':
            self.value = -10000
        else:
            self.value = None
            self.name = 'save_all'       

    def value_update(self, value):
        r"""Update metric value.

        Args:
            value (float): current value for update.
        """
        self.value = value            


def save_checkpoints_update(save_checkpoints, val_metric_logger):
    r"""Update metric for checkpoints.

    Args:
        save_checkpoints (List[CheckpointMetric]): list of optimal metrics for checkpoint.
        val_metric_logger (MetricLogger): validation metric logger.

    Returns:
        List[str]: list of metric names for optimal metics at the current checkpoint.
        List[CheckpointMetric]: list of optimal metrics for checkpoint.
    """
    checkpoint_keys = []
    for checkpoint in save_checkpoints:
        key = checkpoint.name
        keys = list(val_metric_logger.meters.keys())
        if key not in keys + ['save_all']:
            raise ValueError(f"Checkpoint key = {key} is not in {keys}")
        if checkpoint.mode == 'all':
            checkpoint_keys.append(key)
        else:
            val_value = val_metric_logger.meters[key].global_avg
            if checkpoint.mode == 'min':
                flag = checkpoint.value > val_value
            elif checkpoint.mode == 'max':
                flag = checkpoint.value < val_value
            if flag:
                checkpoint_keys.append(key)
                checkpoint.value_update(val_value)
    return checkpoint_keys, save_checkpoints
                