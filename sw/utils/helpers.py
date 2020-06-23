import os
import yaml


#
# I/O
#
def read_yaml(path,*key_path):
    """ read yaml file
    path<str>: path to yaml file
    *key_path: keys to go to in object
    """    
    with open(path,'rb') as file:
        obj=yaml.safe_load(file)
    for k in key_path:
        obj=obj[k]
    return obj


def read_json(path,*key_path):
    """ read json file
    path<str>: path to json file
    *key_path: keys to go to in object
    """    
    with open(path,'rb') as file:
        obj=json.load(file)
    for k in key_path:
        obj=obj[k]
    return obj



#
# UTILS
#
class StrideManager(object):


    def __init__(self,output_stride):
        self.output_stride=output_stride
        self.reset()


    def step(self):
        self.stride_index+=1
        self.current_output_stride=(2**self.stride_index)
        if self.current_output_stride>=self.output_stride:
            self.is_strided=False
            self.dilation_index+=1
            self.dilation_rate=(2**(self.dilation_index))
            self.strides=1


    def reset(self):
        self.stride_index=0
        self.dilation_index=0
        self.dilation_rate=1
        self.strides=2
        self.current_output_stride=1
        self.is_strided=True

