from pprint import pprint
import tensorflow as tf
from tensorflow.keras import layers
from . import xception as xcpt
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


#
# Decoder: a flexible generic decoder
# 
class Decoder(tf.keras.Model):
    #
    # CONSTANTS
    #
    DEFAULT_KEY='decode_256_f128-64_res'
    DEFAULTS=load.config(cfig='decoder',key_path=DEFAULT_KEY)
    UPSAMPLE_MODE='bilinear'
    BEFORE_UP='before'
    AFTER_UP='after'
    SEGMENT='segment'


    #
    # STATIC
    #
    @staticmethod
    def from_config(
            cfig='decoder',
            key_path=DEFAULT_KEY,
            is_file_path=False,
            cfig_dir=load.TFBOX,
            **kwargs):
        config=load.config(
            cfig=cfig,
            key_path=key_path,
            cfig_dir=cfig_dir,
            is_file_path=is_file_path,
            **kwargs)
        print('Decoder:')
        pprint(config)
        return Decoder(**config)


    def __init__(self,
            output_size,
            nb_classes=None,
            input_reducer=DEFAULTS.get('input_reducer'),
            skip_reducers=DEFAULTS.get('skip_reducers'),
            refinements=DEFAULTS.get('refinements'),
            upsample_mode=DEFAULTS.get('upsample_mode'),
            classifier_type=DEFAULTS.get('classifier_type',SEGMENT),
            classifier_position=DEFAULTS.get('classifier_position',BEFORE_UP),
            classifier_kernel_size_list=DEFAULTS.get('classifier_kernel_size_list'),
            classifier_filters_list=DEFAULTS.get('classifier_filters_list'),
            classifier_act=DEFAULTS.get('classifier_act',True),
            classifier_act_config=DEFAULTS.get('classifier_act_config',{}),            
            name=DEFAULTS.get('name',None),
            named_layers=DEFAULTS.get('named_layers',True),
            **kwargs):
        super(Decoder, self).__init__()
        # set properties
        self._upsample_scale=None
        self.output_size=output_size
        self.model_name=name
        self.named_layers=named_layers
        self.upsample_mode=upsample_mode or Decoder.UPSAMPLE_MODE
        self.classifier_position=classifier_position
        # decoder
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
        # classifier
        if classifier_type==Decoder.SEGMENT:
            self.classifier=blocks.SegmentClassifier(
                nb_classes=nb_classes,
                filters_list=classifier_filters_list,
                kernel_size_list=classifier_kernel_size_list,
                output_act=classifier_act,
                output_act_config=classifier_act_config,
                name=self._layer_name('classifier'),
                named_layers=self.named_layers)
        elif not classifier_type:
            self.classifier_position=False
            self.classifier=None
        else:
            raise NotImplementedError(f'{classifier_type} is not a valid classifier')


    def __call__(self,inputs,skips=[],training=False):
        if (skips is None) or (skips is False):
            skips=[]

        x=self._conditional(inputs,self.input_reducer)

        for i,skip in skips:
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
            mode=self.upsample_mode)

        x=self._conditional(
            x,
            self.classifier,
            test=self.classifier_position==Decoder.AFTER_UP)
        return x


    #
    # INTERNAL
    #
    def _layer_name(self,group=None,index=None):
        return blocks.layer_name(
            *[self.model_name,group],
            index=index,
            named=self.named_layers)


    def _named(self,config,group,index=None):
        if self.named_layers:
            config['name']=self._layer_name(group,index=index)
            config['named_layers']=True
        return config


    def _reducer(self,config,index=None):
        if isinstance(config,int):
            filters=config
            config=REDUCER_CONFIG.copy()
            config['filters']=filters
            config=self._named(config,'reducer',index=index)
            return blocks.CBAD(**config)


    def _refinement(self,config,index=None):
        if isinstance(config,int):
            filters=config
            config=REFINEMENT_CONFIG.copy()
            config['filters']=filters
            config=self._named(config,'refinements',index=index)
            return blocks.CBADStack(**config)


    def _scale(self,x,like):
        if not self._upsample_scale:
            self._upsample_scale=self.output_size/x.shape[-2]
        return self._upsample_scale


    def _conditional(self,x,action,index=None,test=True):
        if action and test:
            if index: 
                action=action[index]
            x=action(x)
        return x
