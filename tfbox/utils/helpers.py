import os
import re
import math
import yaml
import tensorflow.keras as keras


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
# PYTHON
#
_SNAKE_RGX=re.compile('((?<=[a-z0-9])[A-Z]|(?!^)[A-Z](?=[a-z]))')
def snake(string):
    return _SNAKE_RGX.sub(r'_\1',str(string)).lower()


def camel(string):
    return re.sub('[\ \-\_]','',str(string).title())


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


#
# FINE TUNING
#
def swap_top(model,top,inputs=None,inputs_shape=None,swap_index=-2):
    if inputs is None:
        if inputs_shape is None:
            inputs=model.inputs
        else:
            inputs=keras.Input(shape=inputs_shape)
    model.trainable=False
    if swap_index:
        model=keras.Model(model.inputs, model.layers[swap_index].output)
    return keras.Model(inputs,top(model(inputs,training=False)))



def match_layer(
        name=None,
        index=None,
        matches=[],
        indices=[],
        searches=[],
        excludes=[]):
    match=False
    if name:
        if name in matches:
            match=True
        else:
            found=next((s for s in searches if s in name),False)
            if found:
                exclude=next((e for e in excludes if e in name),False)
                if not exclude:
                    match=True
    if index and indices:
        match=index in indices
    return match
    

def set_trainable(
        model,
        matches=[],
        indices=[],
        searches=[],
        excludes=[],
        trainable=True,
        return_matched=False,
        noisy=False):
    if matches or indices or searches:
        matches=_as_list(matches)
        indices=_as_list(indices)
        searches=_as_list(searches)
        excludes=_as_list(excludes)
        indices=[int(i) for i in indices]
        if return_matched:
            matched=[]
        for i,l in enumerate(model.layers):
            match=match_layer(
                name=l.name,
                index=i,
                matches=matches,
                indices=indices,
                searches=searches,
                excludes=excludes)
            l.trainable=bool(trainable==match)
            if return_matched and match:
                matched.append(l)
            if noisy and match:
                print(l.name,l.trainable)
        if return_matched:
            return matched


#
# CONSTANTS
#
COLORS=[
    '#8dd3c7',
    '#ffffb3',
    '#bebada',
    '#fb8072',
    '#80b1d3',
    '#fdb462',
    '#b3de69',
    '#fccde5',
    '#d9d9d9',
    '#bc80bd',
    '#ccebc5',
    '#ffed6f',
    '#e41a1c',
    '#377eb8',
    '#4daf4a',
    '#984ea3',
    '#ff7f00',
    '#ffff33',
    '#a65628',
    '#f781bf',
    '#999999',
    '#66c2a5',
    '#fc8d62',
    '#8da0cb',
    '#e78ac3',
    '#a6d854',
    '#ffd92f',
    '#e5c494',
    '#b3b3b3'
]

#
# INTERNAL
#
def _as_list(value):
    if value in [None,False]:
        value=[]
    elif not isinstance(value,list):
        value=[value]
    return value










