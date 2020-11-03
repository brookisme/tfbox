from pprint import pprint
import tensorflow as tf
from tensorflow.keras import layers
from . import base
from . import blocks
from . import load
#
# CONSTANTS
#
REDUCER_CONFIG={
    'kernel_size': 1,
}
REFINEMENT_CONFIG={
    'kernel_size': 3,
    'depth': 2,
    'residual': True
}
OUTPUT_CONV_CONFIG={
    'kernel_size': 3,
    'depth': 1,
    'residual': False,
    'act': True
}
BAND_AXIS=-1
SAFE_RESCALE_ERROR=(
    'tfbox.Decoder: '
    'output rescale leads to fractional value. '
    'use safe_rescale=False to force' )

#
# Decoder: a flexible generic decoder
# 
class Decoder(base.Model):
    #
    # CONSTANTS
    #
    NAME='Decoder'
    UPSAMPLE_MODE='bilinear'
    BEFORE_UP='before'
    AFTER_UP='after'


    def __init__(self,
            nb_classes=None,
            model_config=NAME,
            output_size=None,
            output_ratio=None,
            output_conv=None,
            output_conv_position=None,
            safe_rescale=True,
            key_path='decode_256_f128-64_res',
            is_file_path=False,
            cfig_dir=load.TFBOX,
            classifier_config={
                'classifier_type': base.Model.SEGMENT,
            },
            name=NAME,
            named_layers=True,
            noisy=True):
        super(Decoder, self).__init__(
            name=name,
            named_layers=named_layers,
            noisy=noisy,
            nb_classes=nb_classes,
            classifier_config=classifier_config)
        if isinstance(model_config,str):
            model_config=load.config(
                    cfig=model_config,
                    key_path=key_path,
                    is_file_path=is_file_path,
                    cfig_dir=cfig_dir,
                    noisy=noisy )
        # parse config
        self.model_config=model_config
        self._output_size=self._value(output_size,'output_size')
        self._output_ratio=self._value(output_ratio,'output_ratio',1)
        output_conv=self._value(output_conv,'output_conv')
        self.output_conv_position=self._value(
            output_conv_position,
            'output_conv_position',
            Decoder.AFTER_UP)
        self.safe_rescale=self._value(safe_rescale,'safe_rescale',True)
        input_reducer=model_config.get('input_reducer')
        skip_reducers=model_config.get('skip_reducers')
        refinements=model_config.get('refinements')
        self.upsample_mode=model_config.get('upsample_mode',Decoder.UPSAMPLE_MODE)
        classifier_config=model_config.get('classifier',False)
        # decoder
        self._upsample_scale=None
        self.input_reducer=self._reducer(input_reducer)
        self.output_conv=self._output_conv(output_conv,nb_classes)
        if skip_reducers:
            self.skip_reducers=[
                self._reducer(r,index=i) for i,r in enumerate(skip_reducers) ]
        else:
            self.skip_reducers=None
        if refinements:
            self.refinements=[
                self._refinement(r,index=i) for i,r in enumerate(refinements) ]
        else:
            self.refinements=None


    def set_output(self,like):
        if not self._output_size:
            self._output_size=self._output_rescale(like.shape[-2])


    def __call__(self,inputs,skips=[],training=False):
        if (skips is None) or (skips is False):
            skips=[]
        x=self._conditional(inputs,self.input_reducer)
        skips.reverse()
        for i, skip in enumerate(skips):
            skip=skips[i]
            x=blocks.upsample(x,like=skip,mode=self.upsample_mode)
            skip=self._conditional(skip,self.skip_reducers,index=i)
            x=tf.concat([x,skip],axis=BAND_AXIS)
            x=self._conditional(x,self.refinements,index=i)
        x=self._conditional(
            x,
            self.output_conv,
            test=self.output_conv_position==Decoder.BEFORE_UP)
        x=blocks.upsample(
            x,
            scale=self._scale(x,inputs),
            mode=self.upsample_mode,
            allow_identity=True)
        x=self._conditional(
            x,
            self.output_conv,
            test=self.output_conv_position==Decoder.AFTER_UP)
        return self.output(x)



    #
    # INTERNAL
    #
    def _named(self,config,group,index=None):
        if self.named_layers:
            config['name']=self.layer_name(group,index=index)
            config['named_layers']=True
        return config


    def _reducer(self,config,index=None):
        return self._configurable_block(
            config,
            REDUCER_CONFIG,
            'Conv',
            'reducer',
            index=index)


    def _refinement(self,config,index=None):
        return self._configurable_block(
            config,
            REFINEMENT_CONFIG,
            'Stack',
            'refinements',
            index=index)


    def _output_conv(self,config,nb_classes):
        if config is None:
            config=nb_classes
        return self._configurable_block(
            config,
            OUTPUT_CONV_CONFIG,
            'Stack',
            'output_conv')


    def _configurable_block(self,
            config,
            default_config,
            default_btype,
            root_name,
            index=None):
        if config:
            if isinstance(config,int):
                filters=config
                config=default_config.copy()
                config['filters']=filters
            config=config.copy()
            config=self._named(config,root_name,index=index)
            btype=config.pop('block_type',default_btype)
            return blocks.get(btype)(**config)


    def _output_rescale(self,value):
        fval=value*self._output_ratio
        val=round(fval)
        if (not self.safe_rescale) or (val==fval):
            return val
        else:
            raise ValueError(SAFE_RESCALE_ERROR)


    def _scale(self,x,like):
        if not self._upsample_scale:
            self._upsample_scale=self._output_size/x.shape[-2]
        return self._upsample_scale


    def _conditional(self,x,action,index=None,test=True):
        if action and test:
            if index is not None: 
                action=action[index]
            x=action(x)
        return x


    def _value(self,value,key,default=None):
        if value is None:
            value=self.model_config.get(key,default)
        return value


