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
    DEFAULT_CLASSIFIER=base.Model.SEGMENT
    UPSAMPLE_MODE='bilinear'
    BEFORE_UP='before'
    AFTER_UP='after'


    def __init__(self,
            config,
            file_name=None,
            folder=load.TFBOX,
            nb_classes=None,
            from_logits=None,
            add_classifier=None,
            name=NAME,
            named_layers=True,
            noisy=True):
        super(Decoder, self).__init__(
            name=name,
            named_layers=named_layers,
            noisy=noisy)
        self.config=load.config(
            config,
            file_name or Decoder.NAME,
            folder)
        if add_classifier is None:
            add_classifier=self.config.get('classifier',False)
        # parse config
        self._output_size=self.config.get('output_size')
        self._output_ratio=self.config.get('output_ratio',1)
        self.output_conv_position=self.config.get(
            'output_conv_position',
            Decoder.AFTER_UP)
        self.safe_rescale=self.config.get('safe_rescale',True)
        input_reducer=self.config.get('input_reducer')
        skip_reducers=self.config.get('skip_reducers')
        refinements=self.config.get('refinements')
        self.upsample_mode=self.config.get(
            'upsample_mode',
            Decoder.UPSAMPLE_MODE)
        # decoder
        self._upsample_scale=None
        self.input_reducer=self._reducer(input_reducer)
        self.output_conv=self._output_conv(
            self.config.get('output_conv'),
            nb_classes)
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
        if add_classifier and nb_classes:
            self.set_classifier(
                nb_classes,
                self.config.get('classifier'),
                folder=folder,
                from_logits=from_logits)


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
        if isinstance(config,dict):
            config['filters']=config.get('filters') or nb_classes
        elif config is None:
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
            value=self.config.get(key,default)
        return value


