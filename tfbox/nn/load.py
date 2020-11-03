import os
import re
from pprint import pprint
import tfbox.utils.helpers as h
from pathlib import Path
_nn_dir=Path(__file__).parent
_cwd=Path.cwd()
#
# CONSTANTS
#
TFBOX='tfbox'
LOCAL='local'
CONFIGS_DIR=f'{_nn_dir}/configs'
LOCAL_CONFIGS_DIR=f'{_cwd}/nn/configs'
YAML_RGX='.(yaml|yml)$'
JSON_RGX='.json$'



#
# I/0
#
def config(
        config,
        file_name=None,
        folder=TFBOX,
        noisy=False,
        **kwargs):
    """ load model configs
    Args:
        - config<str|dict>:

            <str>: 
                - relative file path if '.yaml' ext
                - otherwise: comma/dot separated string

                    * c.d.e     => key_path='c.d.e'  

                    * b,c.d.e   => file_name='b.yaml'
                                   key_path='c.d.e'
                    
                    * a,b,c.d.e => folder='a'
                                   file_name='b.yaml'
                                   key_path='c.d.e'
                    
                    where key path c.d.e extracts `yaml_dict[c][d][e]` 
                    from the yaml-dictionary. note: c,d,e will be 
                    converted from camel to snake case.

            <dict>: 
                - if dict contains 'key_path':
                    extract key_path, file_name, folder from dict
                - otherwise: dict is config dictionary

        - file_name<str|None>: a default file_name
        - folder<str|None|bool>:
            - tfbox,None,True: use tfbox config file (tfbox.nn.configs)
            - False or string-false: current working directory
            - Otherwise: directory path

        - kwargs: key-value args to update the loaded config
    """
    if isinstance(config,str):
        if re.search(YAML_RGX,config):
            path=config
        else:
            path, key_path=parse_config_string(config,file_name,folder)           
    else:
        if 'key_path' in config:
            key_path=config['key_path']
            path=config_path(
                config.get('file_name',file_name),
                config.get('folder',folder))
        else:
            path=None
            key_path=None
    if path:
        config=h.read_yaml(path)
    if key_path:
        if isinstance(key_path,str):
            key_path=h.snake(key_path).split('.')
        for k in key_path: config=config[k]
    if kwargs:
        config.update(kwargs)
        pprint(config)
    return config


def parse_config_string(config_string,file_name=None,folder=TFBOX):
    # get key_path,file_name,folder
    parts=config_string.split(',')
    nb_parts=len(parts)
    if nb_parts==3:
        folder=parts[0] or folder
        file_name=parts[1] or file_name
    elif nb_parts==2:
        file_name=parts[0] or file_name
    key_path=parts[-1]
    return config_path(file_name,folder), key_path


def config_path(file_name=None,folder=TFBOX):
    # process file_name/folder
    if not re.search(YAML_RGX,file_name):
        file_name=f'{h.snake(file_name)}.yaml'
    if (folder==TFBOX) or h.truey(folder):
        folder=CONFIGS_DIR
    elif (folder==LOCAL):
        folder=LOCAL_CONFIGS_DIR
    elif h.falsey(folder):
        folder=_cwd
    # get path
    if h.noney(folder):
        path=file_name
    else:
        path=f'{folder}/{file_name}'
    return path


