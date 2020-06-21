import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import activations
#
# CONSTANTS
#
DEFAULT_ACTIVATION='relu'



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
def segment_classifier(out_ch,kernels=[3],act=None,act_config={},dilation_rate=1):
    print('OUT',out_ch,dilation_rate,kernels)
    def _block(x,act=act,out_ch=out_ch):
        for k in kernels[:-1]: 
            print('conv',k,out_ch)
            x=CBAD(out_ch,kernel_size=k,dilation_rate=dilation_rate)(x)
        print('conv',kernels[-1],out_ch)
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



DEFAULT_DROPOUT_RATE=0.5
class CBAD(tf.keras.Model):
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
            dropout=False,
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
        if batch_norm:
            self.bn=layers.BatchNormalization()
        else:
            self.bn=False
        if act:
            self.act=get_activation(act,**act_config)
        else:
            self.act=False
        if dropout:
            if dropout is True:
                dropout=DEFAULT_DROPOUT_RATE
            self.do=layers.Dropout(rate,**dropout_config)
        else:
            self.do=False
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






def sepres(name,filters,filters_in=None,dilation_rate=1,strides=2):
    if not filters_in:
        filters_in=filters
    print('SEPRES',filters_in,dilation_rate)
    def _block(x):
        print('SEPRES1',filters_in,dilation_rate)

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