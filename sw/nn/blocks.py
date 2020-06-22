import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras import activations
#
# CONSTANTS
#
DEFAULT_ACTIVATION='relu'
DEFAULT_DROPOUT=False
DEFAULT_DROPOUT_RATE=0.5
DEFAULT_MAX_POOLING={
    'pool_size': 2, 
    'strides': 2, 
    'padding': "same"
}


#
# HELPERS
#
ACTIVATIONS={
    'relu': layers.ReLU,
    'sigmoid': activations.sigmoid,
    'softmax': layers.Softmax
}

def get_activation(act,**config):
    if act:
        if act is True:
            act=DEFAULT_ACTIVATION
        if isinstance(act,str):
            act=ACTIVATIONS[act.lower()]
        act=act(**config)
    return act



#
# GENERAL BLOCKS 
#
class CBAD(keras.Model):
    """ Conv-BatchNorm-Activation-Dropout
    """
    #
    # PUBLIC
    #
    def __init__(self,
            filters,
            kernel_size=3,
            padding='same',
            seperable=False,
            batch_norm=True,
            act=True,
            act_config={},
            dilation_rate=1,
            strides=1,
            dropout=DEFAULT_DROPOUT,
            dropout_config={},
            act_last=False,
            **conv_config):
        super(CBAD, self).__init__()
        if seperable:
            _conv=layers.SeparableConv2D
        else:
            _conv=layers.Conv2D
        if dilation_rate>1:
            strides=1
        if kernel_size in [1,(1,1)]:
            dilation_rate=1
        self.conv=_conv(
            filters=filters,
            kernel_size=kernel_size,
            padding=padding,
            dilation_rate=dilation_rate,
            strides=strides,
            **conv_config)
        self.bn=self._batch_norm(batch_norm)
        self.act=self._activation(act,act_config)
        self.do=self._dropout(dropout,dropout_config)
        self.act_last=act_last


    def __call__(self,x,training=False):
        x=self.conv(x)
        if self.bn: x=self.bn(x)
        if self.act_last:
            if training and self.do: x=self.do(x)
            if self.act: x=self.act(x)
        else:
            if self.act: x=self.act(x)
            if training and self.do: x=self.do(x)            
        return x


    #
    # INTERNAL
    #
    def _batch_norm(self,batch_norm):
        if batch_norm:
            bn=layers.BatchNormalization()
        return bn      


    def _activation(self,act,config):
        if act:
            act=get_activation(act,**config)
        return act


    def _dropout(self,dropout,config):
        if dropout:
            if dropout is True:
                dropout=DEFAULT_DROPOUT_RATE
            dropout=layers.Dropout(rate,**config)
        return dropout







class CBADStack(keras.Model):
    """ (Res)Stack of CBAD Blocks
    """
    #
    # CONSTANTS
    #
    IDENTITY='ident'



    #
    # PUBLIC
    #
    def __init__(self,
            filters=None,
            kernel_size=3,
            depth=1,
            filters_out=None,
            filters_list=None,
            kernel_size_list=None,
            padding='same',
            residual=True,
            residual_act=True,
            seperable=False,
            batch_norm=True,
            act=True,
            act_config={},
            dilation_rate=1,
            strides=1,
            output_stride=1,
            max_pooling=False,
            dropout=DEFAULT_DROPOUT,
            dropout_config={},
            act_last=False,
            **conv_config):
        super(CBADStack, self).__init__()
        if output_stride not in [1,2]:
            raise NotImplemented
        if filters_list:
            depth=len(filters_list)
        elif kernel_size_list:
            depth=len(kernel_size_list)
        if not filters_list:
            if filters_out is None:
                filters_out=filters
            filters_list=[filters]*(depth-1)+[filters_out]
        if not kernel_size_list:
            kernel_size_list=[kernel_size]*depth
        self.filters_list=filters_list
        self.kernel_size_list=kernel_size_list
        self._set_config(
            act,
            output_stride,
            dilation_rate,
            max_pooling,
            seperable=seperable,
            batch_norm=batch_norm,
            padding=padding,
            act_config=act_config,
            **conv_config)
        self.residual=self._residual(
            residual,
            residual_act,
            filters_list[-1],
            output_stride)
        self.stack=self._build_stack(output_stride)
            

    def __call__(self,x,training=False,**kwargs):
        if self.residual==CBADStack.IDENTITY:
            res=x
        elif self.residual:
            res=self.residual(x)
        else:
            res=False
        for layer in self.stack:
            x=layer(x)
        if res is not False:
            x=layers.add([res,x])
        return x



    #
    # INTERNAL
    #
    def _set_config(self,
            act,
            output_stride,
            dilation_rate,
            max_pooling,
            **shared_config):
        self.act=act
        self.max_pooling_config=self._max_pooling_config(
            output_stride,
            dilation_rate,
            max_pooling)
        shared_config['dilation_rate']=dilation_rate
        self.shared_config=shared_config



    def _max_pooling_config(self,output_stride,dilation_rate,max_pooling):
        if max_pooling and (output_stride==2) and (dilation_rate==1):
            if max_pooling is True:
                max_pooling=DEFAULT_MAX_POOLING
            return max_pooling
        else:
            return False


    def _residual(self,residual,residual_act,filters,output_stride):
        if residual and (residual!=CBADStack.IDENTITY):
            if not residual_act:
                act=False
            else:
                act=self.act
            residual=CBAD(
                filters=filters,
                kernel_size=1,
                strides=output_stride,
                act=act,
                **self.shared_config)
        return residual


    
    def _build_stack(self,output_stride):
        _layers=[] 
        last_layer_index=len(self.filters_list)-1
        for i,(f,k) in enumerate(zip(self.filters_list,self.kernel_size_list)):
            if (i==last_layer_index) and (not self.max_pooling_config):
                strides=output_stride
            else:
                strides=1
            _layers.append(CBAD(
                filters=f,
                kernel_size=k,
                strides=strides,
                act=self.act,
                **self.shared_config))
            if (i==last_layer_index) and self.max_pooling_config:
                _layers.append(layers.MaxPooling2D(**self.max_pooling_config))
        return _layers







#
# PRE-CONFIGURED BLOCKS
#
class SegmentClassifier(keras.Model):
    """
    """
    #
    # CONSTANTS
    #
    AUTO='auto'



    #
    # PUBLIC
    #
    def __init__(self,
            nb_classes,
            depth=1,
            filters=None,
            filters_list=None,
            kernel_size=3,
            kernel_size_list=None,
            output_act=None,
            output_act_config={},
            seperable_preclassification=False,
            residual_preclassification=False,
            **stack_config):
        super(SegmentClassifier, self).__init__()
        kernel_size_list=self._kernel_size_list(
            kernel_size_list,
            kernel_size,
            depth)
        filters_list=self._filters_list(
            filters_list,
            filters,
            nb_classes,
            len(kernel_size_list))
        if len(filters_list)>1:
            self.preclassifier=CBADStack(
                filters_list=filters_list[:-1],
                kernel_size_list=kernel_size_list[:-1],
                seperable=seperable_preclassification,
                residual=residual_preclassification,
                **stack_config)
        else:
            self.preclassifier=False
        self.classifier=CBAD(
                filters=filters_list[-1],
                kernel_size=kernel_size_list[-1],
                act=self._activation(nb_classes,output_act),
                act_config=output_act_config)



    def __call__(self,x,training=False,**kwargs):
        if self.preclassifier:
            x=self.preclassifier(x)
        return self.classifier(x)



    #
    # INTERNAL
    #
    def _filters_list(self,filters_list,filters,nb_classes,depth):
        if filters_list:
            if filters_list[-1]!=nb_classes:
                raise ValueError('last filters value must equal nb_classes')
        else:
            if filters is None:
                filters=nb_classes
            filters_list=[filters]*(depth-1)+[nb_classes]
        return filters_list


    def _kernel_size_list(self,kernel_size_list,kernel_size,depth):
        if not kernel_size_list:
            kernel_size_list=[kernel_size]*depth
        return kernel_size_list


    def _activation(self,nb_classes,act):
        if (act is None) or (act==SegmentClassifier.AUTO):
            if nb_classes==1:
                act='sigmoid'
            else:
                act='softmax'
        return act


