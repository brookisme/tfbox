import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import sw.nn.blocks as blocks




class Xception(tf.keras.Model):
    #
    # CONSTANTS
    #
    AUTO='auto'



    #
    # PUBLIC
    #
    def __init__(self,
            output_stride=8,
            entry_flow_prefilters_stack=[32,64],
            entry_flow_filters_stack=[128,256,728],
            middle_flow_filters=AUTO,
            middle_flow_depth=16,
            exit_flow_filters_in=AUTO,
            exit_flow_filters=1024,
            exit_flow_postfilters_stack=[1536,1536,2048],
            classifier=False):
        super(Xception, self).__init__()
        self.stride_manager=blocks.StrideManager(output_stride)
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
        x=self._process_stack(self.entry_stack,x)
        skips=[x]
        x=self._process_stack(self.middle_stack,x)
        x=self._process_stack(self.exit_stack,x)
        if self.classifier:
            return self.classifier(x)
        else:
            return x, skips



    #
    # INTERNAL
    #
    def _entry_flow(self,prefilters,filters):
        print('ENTRY')
        _layers=[ blocks.CBAD(filters=prefilters[0],strides=2) ]
        self.stride_manager.step()
        for f in prefilters[1:]:
            _layers.append( blocks.CBAD(filters=f) )
        for f in filters[1:]:
            _layers.append(blocks.CBADStack(
                    seperable=True,
                    filters=f,
                    dilation_rate=self.stride_manager.dilation_rate,
                    output_stride=self.stride_manager.strides ))
            self.stride_manager.step()
        return _layers, filters[-1]


    def _middle_flow(self,filters,flow_depth,prev_filters):
        print('MIDDLE')
        _layers=[]
        if filters==Xception.AUTO:
            filters=prev_filters
        for _ in range(flow_depth):
            _layers.append(blocks.CBADStack(
                seperable=True,
                filters=filters,
                dilation_rate=self.stride_manager.dilation_rate,
                depth=flow_depth,
                residual=blocks.CBADStack.IDENTITY ))
        return _layers, filters


    def _exit_flow(self,filters_in,filters,postfilters,prev_filters):
        print('EXIT')
        if filters_in==Xception.AUTO:
            filters_in=prev_filters
        _layers=[]
        _layers.append(blocks.CBADStack(
                seperable=True,
                filters=filters,
                filters_in=filters_in,
                dilation_rate=self.stride_manager.dilation_rate,
                output_stride=self.stride_manager.strides ))
        self.stride_manager.step()
        for f in postfilters:
            _layers.append( blocks.CBAD(filters=f) )
        if postfilters:
            filters=postfilters[-1]
        else:
            filters=filters[-1]
        return _layers, filters


    def _process_stack(self,stack,x):
        for layer in stack:
            x=layer(x)
        return x



