import os
import re
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
        cfig,
        key_path=None,
        is_file_path=False,
        cfig_dir=TFBOX,
        **kwargs):
    if isinstance(cfig,str):
        if not is_file_path:
            parts=cfig.split('.')
            name=parts[0]
            key_path=key_path or parts[1:] or name
            if isinstance(key_path,str):
                key_path=[key_path]
            cfig=f'{name}.yaml'
            if cfig_dir in [TFBOX,None,True]:
                cfig_dir=CONFIGS_DIR
            if cfig_dir:
                cfig=f'{cfig_dir}/{name}.yaml'
        cfig=h.read_yaml(cfig)
    if key_path:
        for k in key_path: cfig=cfig[k]
    cfig.update(kwargs)
    return cfig





