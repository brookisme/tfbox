import os
import re
from pprint import pprint
import tfbox.utils.helpers as h
_nn_dir=os.path.dirname(os.path.realpath(__file__))
#
# CONSTANTS
#
TFBOX='tfbox'
CONFIGS_DIR=f'{_nn_dir}/configs'
YAML_RGX='.(yaml|yml)$'
JSON_RGX='.json$'





#
# I/0
#
def config(
        cfig=None,
        key_path=None,
        cfig_dir=TFBOX,
        is_file_path=False,
        config_string=None,
        noisy=False,
        **kwargs):
    """ load model configs
    Args:
        - cfig<str|dict|None>:
            * <str>: 
                - name (w/o '.yaml' ext) of config file
                - file_path (see is_file_path below)
            * <dict>: model config as dict
            * None: requires config_string
        - key_path<str>: 
            dot-path for key in cfig-file containing the config
        - cfig_dir<str>:
            - tfbox,None,True: use tfbox config file (tfbox.nn.configs)
            - False: current working directory
            - Otherwise: directory path
        - is_file_path<bool>:
            - if true: cfig arg is the path to the config file
        - config_string<str|None>:
            - if config_string derive above args using `parse_config_string`
        - kwargs: key-value args to update the loaded config
    """
    if config_string:
        cfig, key_path, is_file_path=parse_config_string(
            config_string,
            cfig=cfig)
    if isinstance(cfig,str):
        title=cfig
        cfig=h.snake(cfig)
        if not is_file_path:
            if cfig_dir in [TFBOX,None,True]:
                cfig_dir=CONFIGS_DIR
            if cfig_dir:
                cfig=f'{cfig_dir}/{cfig}'
        cfig=h.read_yaml(f'{cfig}.yaml')
    else:
        title='config'
    if key_path:
        key_path=h.snake(key_path)
        if isinstance(key_path,str):
            key_path=key_path.split('.')
        for k in key_path: cfig=cfig[k]
    if kwargs:
        if isinstance(cfig,dict):
            cfig.update(kwargs)
        else:
            raise ValueError('kwargs only allowed for key-value configs.')
    if noisy:
        pprint(cfig)
    return cfig



def parse_config_string(config_string,cfig=None):
    key_path=is_file_path=None
    parts=config_string.split(':')
    if cfig:
        parts=[cfig]+parts
    nb_parts=len(parts)
    cfig=parts[0]
    if nb_parts>1:
        key_path=parts[1]
        if nb_parts>2:
            is_file_path=str(parts[2]).lower()=='true'
    return cfig, key_path, is_file_path




