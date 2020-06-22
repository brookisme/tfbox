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
# BLOCKS
#
class CBAD(keras.Model):
    """ Conv-BatchNorm-Activation-Dropout

    """
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

    IDENTITY='ident'

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
        if not filters_list:
            if filters_out:
                filters_list=[filters]*(depth-1)+[filters]
            else:         
                filters_list=[filters]*depth
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



def segment_classifier(out_ch,kernels=[3],act=None,act_config={},dilation_rate=1):
    def _block(x,act=act,out_ch=out_ch):
        for k in kernels[:-1]: 
            x=CBAD(out_ch,kernel_size=k,dilation_rate=dilation_rate)(x)
        x=CBAD(out_ch,kernel_size=kernels[-1],dilation_rate=dilation_rate,act=False)(x)
        if act is None:
            if out_ch==1:
                act='sigmoid'
            else:
                act='softmax'
        if act:
            x=get_activation(act,**act_config)(x)
        return x
    return _block




def sepres(name,filters,filters_in=None,dilation_rate=1,strides=2):
    if not filters_in:
        filters_in=filters
    def _block(x):
        res=CBAD(
            name=f'{name}-res',
            seperable=True,
            filters=filters,
            kernel_size=1,
            strides=strides,
            dilation_rate=dilation_rate)(x)
        x=CBAD(
            name=f'{name}-b1',
            seperable=True,
            filters=filters_in,
            kernel_size=3,
            strides=1,
            dilation_rate=dilation_rate)(x)
        x=CBAD(
            name=f'{name}-b2',
            seperable=True,
            filters=filters,
            kernel_size=3,
            strides=1,
            dilation_rate=dilation_rate)(x)
        x=CBAD(
            name=f'{name}-b3',
            seperable=True,
            filters=filters,
            kernel_size=3,
            strides=strides,
            dilation_rate=dilation_rate)(x)
        return layers.add([res, x])
    return _block

