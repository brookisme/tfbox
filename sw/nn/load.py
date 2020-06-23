import os
import re
import sw.utils.helpers as h
_nn_dir=os.path.dirname(os.path.realpath(__file__))
#
# CONSTANTS
#
CONFIGS_DIR=f'{_nn_dir}/configs'
YAML_RGX='.(yaml|yml)$'
JSON_RGX='.json$'



#
# I/0
#
def config(cfig,key_path=None,is_file_path=False,**kwargs):
    if isinstance(cfig,str):
        if not is_file_path:
            parts=cfig.split('.')
            name=parts[0]
            key_path=key_path or parts[1:]
            if isinstance(key_path,str):
                key_path=[key_path]
            cfig=f'{CONFIGS_DIR}/{name}.yaml'
        if re.search(JSON_RGX,cfig):
            cfig=h.read_json(cfig)
        else:
            cfig=h.read_yaml(cfig)
    if key_path:
        for k in key_path: cfig=cfig[k]
    cfig.update(kwargs)
    return cfig





