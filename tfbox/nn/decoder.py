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
            output_size=None,
            output_ratio=None,
            safe_rescale=True,
            model_config=NAME,
            key_path='decode_256_f128-64_res',
            is_file_path=False,
            cfig_dir=load.TFBOX,
            classifier_position=AFTER_UP,
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
        self._output_size=output_size or model_config.get('output_size')
        self._output_ratio=output_ratio or model_config.get('output_ratio',1)
        self.safe_rescale=safe_rescale or model_config.get('safe_rescale',True)
        input_reducer=model_config.get('input_reducer')
        skip_reducers=model_config.get('skip_reducers')
        refinements=model_config.get('refinements')
        self.upsample_mode=model_config.get('upsample_mode',Decoder.UPSAMPLE_MODE)
        classifier_config=model_config.get('classifier',False)
        # decoder
        self._upsample_scale=None
        self.input_reducer=self._reducer(input_reducer)
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
        self.classifier_position=classifier_position


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
            self.classifier,
            test=self.classifier_position==Decoder.BEFORE_UP)
        x=blocks.upsample(
            x,
            scale=self._scale(x,inputs),
            mode=self.upsample_mode,
            allow_identity=True)
        x=self._conditional(
            x,
            self.classifier,
            test=self.classifier_position==Decoder.AFTER_UP)
        return x


    #
    # INTERNAL
    #
    def _named(self,config,group,index=None):
        if self.named_layers:
            config['name']=self.layer_name(group,index=index)
            config['named_layers']=True
        return config


    def _reducer(self,config,index=None):
        if isinstance(config,int):
            filters=config
            config=REDUCER_CONFIG.copy()
            config['filters']=filters
        config=config.copy()
        config=self._named(config,'reducer',index=index)
        btype=config.pop('block_type','Conv')
        return blocks.get(btype)(**config)


    def _refinement(self,config,index=None):
        if isinstance(config,int):
            filters=config
            config=REFINEMENT_CONFIG.copy()
            config['filters']=filters
        config=config.copy()
        config=self._named(config,'refinements',index=index)
        btype=config.pop('block_type','Stack')
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
