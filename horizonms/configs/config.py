import yaml
import collections.abc
import copy
from typing import Dict


def read_config_file(file_name: str):
    r"""Read configuration file.

    Args:
        file_name (str): name of the configuration file.

    Returns: 
        Dict: configuration in dictionary format.
    """
    with open(file_name) as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
    return data


def save_config_file(file_name: str, data_dict: Dict):
    r"""Save configuration file.

    Args:
        file_name (str): name of the configuration file.
        data_dict (Dict): configuration for saving.
    """
    with open(file_name, 'w') as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)


def config_updates(config: Dict, config_new: Dict):
    r"""Update configuration.

    Args:
        config (Dict): configuration to be updated.
        config_new (Dict): new configuration as dictionary for update.

    Returns:
        config_out (Dict): updated configration.
    """
    config_out = copy.deepcopy(config)
    for k, v in config_new.items():
        if isinstance(v, collections.abc.Mapping):
            config_out[k] = config_updates(config_out.get(k, {}), v)
        else:
            config_out[k] = v
    return config_out
