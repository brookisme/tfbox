from pprint import pprint
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tfbox.utils.helpers import StrideManager
from . import base
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
    DEFAULT_KEY='xlight_os8_mf16'
    DEFAULT_SEGEMENT_KEY='xlight_os8_mf16_seg'
    DEFAULTS=load.config(cfig='xception',key_path=DEFAULT_KEY)
    GAP='gap'
    SEGMENT='segment'


    #
    # STATIC
    #
    @staticmethod
    def from_config(
            cfig='xception',
            key_path=DEFAULT_KEY,
            cfig_dir=load.TFBOX,
            is_file_path=False,
            **kwargs):
        config=load.config(
            cfig=cfig,
            key_path=key_path,
            cfig_dir=cfig_dir,
            is_file_path=is_file_path,
            **kwargs)
        print('XCEPTION:')
        pprint(config)
        return Xception(**config)


    #
    # PUBLIC
    #
    def __init__(self,
            output_stride=DEFAULTS['output_stride'],
            dropout=DEFAULTS.get('dropout',False),
            entry_flow_prestrides=DEFAULTS['entry_flow_prestrides'],
            entry_flow_prefilters_stack=DEFAULTS['entry_flow_prefilters_stack'],
            entry_flow_filters_stack=DEFAULTS['entry_flow_filters_stack'],
            entry_flow_strides_stack=DEFAULTS.get('entry_flow_strides_stack'),
            entry_flow_seperable=DEFAULTS.get('entry_flow_seperable',True),
            entry_flow_dropout=DEFAULTS.get('entry_flow_dropout',False),
            middle_flow_filters=DEFAULTS['middle_flow_filters'],
            middle_flow_depth=DEFAULTS['middle_flow_depth'],
            middle_flow_seperable=DEFAULTS.get('middle_flow_seperable',True),
            middle_flow_dropout=DEFAULTS.get('middle_flow_dropout',False),
            exit_flow_filters_in=DEFAULTS['exit_flow_filters_in'],
            exit_flow_filters=DEFAULTS['exit_flow_filters'],
            exit_flow_seperable=DEFAULTS.get('exit_flow_seperable',True),
            exit_flow_postfilters_stack=DEFAULTS['exit_flow_postfilters_stack'],
            exit_flow_dropout=DEFAULTS.get('exit_flow_dropout',False),
            nb_classes=DEFAULTS.get('nb_classes',None),
            classifier_type=DEFAULTS.get('classifier_type',False),
            classifier_act=DEFAULTS.get('classifier_act',True),
            classifier_act_config=DEFAULTS.get('classifier_act_config',{}),
            classifier_kernel_size_list=DEFAULTS.get('classifier_kernel_size_list'),
            classifier_filters_list=DEFAULTS.get('classifier_filters_list'),
            keep_mid_step=DEFAULTS.get('keep_mid_step',True),
            skip_indices=DEFAULTS.get('skip_indices',True),
            name=DEFAULTS.get('name',None),
            named_layers=DEFAULTS.get('named_layers',True)):
        super(Xception, self).__init__()
        self.model_name=name
        self.named_layers=named_layers
        if dropout:
           entry_flow_dropout=dropout
           middle_flow_dropout=dropout
           exit_flow_dropout=dropout


        self.stride_manager=StrideManager(
            output_stride,
            keep_mid_step=keep_mid_step,
            keep_indices=skip_indices)

        self.entry_stack, filters_out=self._entry_flow(
            entry_flow_prestrides,
            entry_flow_prefilters_stack,
            entry_flow_filters_stack,
            entry_flow_strides_stack,
            entry_flow_seperable,
            entry_flow_dropout)

        if middle_flow_filters:
            self.middle_stack, filters_out=self._middle_flow(
                middle_flow_filters,
                middle_flow_depth,
                middle_flow_seperable,
                middle_flow_dropout,
                filters_out)
        else:
            self.middle_stack=False
        if exit_flow_filters:
            self.exit_stack, filters_out=self._exit_flow(
                exit_flow_filters_in,
                exit_flow_filters,
                exit_flow_seperable,
                exit_flow_dropout,
                exit_flow_postfilters_stack,
                filters_out)
        else:
            self.exit_stack=False
        if classifier_type==Xception.GAP:
            # todo: global-avg-pooling+dense+classifier
            pass
        elif classifier_type==Xception.SEGMENT:
            self.classifier=blocks.SegmentClassifier(
                nb_classes=nb_classes,
                filters_list=classifier_filters_list,
                kernel_size_list=classifier_kernel_size_list,
                output_act=classifier_act,
                output_act_config=classifier_act_config,
                name=self._layer_name('classifier'),
                named_layers=self.named_layers)
        else:
            self.classifier=False



    def __call__(self,x,training=False,**kwargs):
        x,entry_skips=self._process_stack(self.entry_stack,x)
        if self.middle_stack:
            x=self._process_stack(self.middle_stack,x,False)
        if self.exit_stack:            
            x,exit_skips=self._process_stack(self.exit_stack,x)
        if self.classifier:
            return self.classifier(x)
        else:
            return x, entry_skips+exit_skips



    #
    # INTERNAL
    #
    def _layer_name(self,group=None,index=None):
        return blocks.layer_name(
            *[self.model_name,group],
            index=index,
            named=self.named_layers)


    def _entry_flow(self,prestrides,prefilters,filters,strides,seperable,dropout):
        _layers=[]
        for i,(s,f) in enumerate(zip(prestrides,prefilters)):
            _layers.append(blocks.Conv(
                filters=f,
                strides=self.stride_manager.strides(s),
                dilation_rate=self.stride_manager.dilation_rate,
                dropout=dropout,
                keep_output=self.stride_manager.keep_index,
                name=self._layer_name('entry',i),
                named_layers=self.named_layers))
            self.stride_manager.step(s)
        if not strides:
            strides=[2]*len(filters)
        for j,(f,s) in enumerate(zip(filters,strides)):
            _layers.append(blocks.Stack(
                    seperable=seperable,
                    dropout=dropout,
                    depth=3,
                    filters=f,
                    output_stride=self.stride_manager.strides(s),
                    dilation_rate=self.stride_manager.dilation_rate,
                    keep_output=self.stride_manager.keep_index,
                    name=self._layer_name('entry',i+j),
                    named_layers=self.named_layers))
            self.stride_manager.step(s)
        return _layers, filters[-1]


    def _middle_flow(self,filters,flow_depth,seperable,dropout,prev_filters):
        _layers=[]
        if filters==Xception.AUTO:
            filters=prev_filters
        for i in range(flow_depth):
            _layers.append(blocks.Stack(
                seperable=seperable,
                dropout=dropout,
                depth=3,
                filters=filters,
                dilation_rate=self.stride_manager.dilation_rate,
                residual=blocks.Stack.IDENTITY,
                name=self._layer_name('middle',i),
                named_layers=self.named_layers))
        return _layers, filters


    def _exit_flow(self,filters_in,filters,seperable,dropout,postfilters,prev_filters):
        if filters_in==Xception.AUTO:
            filters_in=prev_filters
        _layers=[]
        _layers.append(blocks.Stack(
                seperable=seperable,
                dropout=dropout,
                depth=3,
                filters=filters,
                filters_in=filters_in,
                output_stride=self.stride_manager.strides(),
                dilation_rate=self.stride_manager.dilation_rate,
                keep_output=False,
                name=self._layer_name('exit',0),
                named_layers=self.named_layers))
        self.stride_manager.step()
        for i,f in enumerate(postfilters):
            _layers.append(blocks.Conv(
                filters=f,
                seperable=seperable,
                dropout=dropout,
                dilation_rate=self.stride_manager.dilation_rate,
                name=self._layer_name('exit',i+1),
                named_layers=self.named_layers))
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
