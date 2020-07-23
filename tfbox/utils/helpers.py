import os
import math
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


    def __init__(self,output_stride,keep_mid_step=False,keep_indices=True):
        self.output_stride=output_stride
        self.strided_steps=round(math.log2(output_stride))
        self._set_keepers(keep_mid_step,keep_indices)
        self.reset()


    def step(self,strides=True):
        if strides is True:
            strides=2
        if (strides>1):
            self.stride_index+=1
            new_output_stride=self.current_output_stride*strides
            if new_output_stride>=self.output_stride:
                self.at_max_stride=True
                self.dilation_index+=1
                self.dilation_rate*=strides
                self.stride_state=1
                self.keep_index=False
            else:
                self.current_output_stride=new_output_stride
                self.keep_index=self._keep_index()


    def strides(self,strides=None):
        if self.at_max_stride:
            return 1
        else:
            if (strides is None) or (strides is True):
                strides=2
            elif strides is False:
                strides=1
            return strides


    def reset(self):
        self.stride_index=0
        self.dilation_index=0
        self.dilation_rate=1
        self.stride_state=2
        self.current_output_stride=1
        self.at_max_stride=False
        self.keep_index=self._keep_index()


    def _keep_index(self):
        if isinstance(self.keep_indices,list):
            return self.stride_index in self.keep_indices
        else:
            return self.keep_indices


    def _set_keepers(self,keep_mid_step,keep_indices):
        if keep_mid_step:
            self.keep_indices=[keep_mid_step]
        else:
            self.keep_indices=keep_indices
        if self.keep_indices is True:
            self.nb_keepers=self.strided_steps
        elif self.keep_indices:
            self.nb_keepers=len(self.keep_indices)
        else:
            self.nb_keepers=0

