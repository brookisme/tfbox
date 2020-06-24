import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sw.utils.helpers import StrideManager
from . import blocks
from . import load
#
#
# CONSTANTS
#




#
# Xception Network:
#
class Xception(tf.keras.Model):
    #
    # CONSTANTS
    #
    AUTO='auto'
    DEFAULT_KEY='small'
    DEFAULTS=load.config(cfig='xception',key_path=DEFAULT_KEY)



    #
    # STATIC
    #
    @staticmethod
    def from_config(
            key_path=DEFAULT_KEY,
            cfig='xception',
            is_file_path=False,
            **kwargs):
        config=load.config(
            cfig=cfig,
            key_path=key_path,
            is_file_path=is_file_path,
            **kwargs)
        return Xception(**config)


    #
    # PUBLIC
    #
    def __init__(self,
            output_stride=DEFAULTS['output_stride'],
            entry_flow_prefilters_stack=DEFAULTS['entry_flow_prefilters_stack'],
            entry_flow_filters_stack=DEFAULTS['entry_flow_filters_stack'],
            middle_flow_filters=DEFAULTS['middle_flow_filters'],
            middle_flow_depth=DEFAULTS['middle_flow_depth'],
            exit_flow_filters_in=DEFAULTS['exit_flow_filters_in'],
            exit_flow_filters=DEFAULTS['exit_flow_filters'],
            exit_flow_postfilters_stack=DEFAULTS['exit_flow_postfilters_stack'],
            classifier=DEFAULTS.get('classifier',False),
            keep_mid_step=DEFAULTS.get('keep_mid_step',True),
            skip_indices=DEFAULTS.get('skip_indices',True)):
        super(Xception, self).__init__()
        self.stride_manager=StrideManager(
            output_stride,
            keep_mid_step=keep_mid_step,
            keep_indices=skip_indices)
        self.entry_stack, filters_out=self._entry_flow(
            entry_flow_prefilters_stack,
            entry_flow_filters_stack)
        self.middle_stack, filters_out=self._middle_flow(
            middle_flow_filters,
            middle_flow_depth,
            filters_out)
        self.exit_stack, filters_out=self._exit_flow(
            exit_flow_filters_in,
            exit_flow_filters,
            exit_flow_postfilters_stack,
            filters_out)
        self.classifier=classifier


    def __call__(self,x,training=False,**kwargs):
        x,entry_skips=self._process_stack(self.entry_stack,x)
        x=self._process_stack(self.middle_stack,x,False)
        x,exit_skips=self._process_stack(self.exit_stack,x)
        if self.classifier:
            return self.classifier(x)
        else:
            return x, entry_skips+exit_skips



    #
    # INTERNAL
    #
    def _entry_flow(self,prefilters,filters):
        _layers=[ blocks.CBAD(
            filters=prefilters[0],
            strides=self.stride_manager.strides,
            dilation_rate=self.stride_manager.dilation_rate,
            keep_output=self.stride_manager.keep_index) ]
        self.stride_manager.step()
        for f in prefilters[1:]:
            _layers.append(blocks.CBAD(
                filters=f,
                dilation_rate=self.stride_manager.dilation_rate) )
        for f in filters:
            _layers.append(blocks.CBADStack(
                    seperable=True,
                    depth=3,
                    filters=f,
                    output_stride=self.stride_manager.strides,
                    dilation_rate=self.stride_manager.dilation_rate,
                    keep_output=self.stride_manager.keep_index ))
            self.stride_manager.step()
        return _layers, filters[-1]


    def _middle_flow(self,filters,flow_depth,prev_filters):
        _layers=[]
        if filters==Xception.AUTO:
            filters=prev_filters
        for _ in range(flow_depth):
            _layers.append(blocks.CBADStack(
                seperable=True,
                depth=3,
                filters=filters,
                dilation_rate=self.stride_manager.dilation_rate,
                residual=blocks.CBADStack.IDENTITY ))
        return _layers, filters


    def _exit_flow(self,filters_in,filters,postfilters,prev_filters):
        if filters_in==Xception.AUTO:
            filters_in=prev_filters
        _layers=[]
        _layers.append(blocks.CBADStack(
                seperable=True,
                depth=3,
                filters=filters,
                filters_in=filters_in,
                output_stride=self.stride_manager.strides,
                dilation_rate=self.stride_manager.dilation_rate,
                keep_output=False ))
        self.stride_manager.step()
        for f in postfilters:
            _layers.append(blocks.CBAD(
                filters=f,
                seperable=True,
                dilation_rate=self.stride_manager.dilation_rate))
        if postfilters:
            filters=postfilters[-1]
        else:
            filters=filters[-1]
        return _layers, filters


    def _process_stack(self,stack,x,return_skips=True):
        if return_skips:
            skips=[]
        for layer in stack:
            x=layer(x)
            if return_skips:
                skips=self._update_skips(layer,skips,x)
        if return_skips:
            return x, skips
        else:
            return x


    def _update_skips(self,layer,skips,x,force_update=False):
        try:
            if force_update or (layer.keep_output): 
                skips.append(x)
        except:
            pass
        return skips
