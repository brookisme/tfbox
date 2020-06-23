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
def config(name,key=None,is_file_path=False,**kwargs):
    if not is_file_path:
        name=f'{CONFIGS_DIR}/{name}'
    if re.search(JSON_RGX,name):
        cfig=h.read_json(name)
    else:
        if not re.search(YAML_RGX,name):
            name=f'{name}.yaml'
        cfig=h.read_yaml(name)
    if key:
        cfig=cfig[key]
    return cfig
