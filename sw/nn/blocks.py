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
def segment_classifier(out_ch,kernels=[3],act=None,act_config={}):
    def _block(x,act=act):
        for k in kernels[:-1]: 
            x=conv(out_ch,kernel_size=k)(x)
        x=conv(out_ch,kernel_size=kernels[-1],act=False)(x)
        if act is None:
            if out_ch==1:
                act='sigmoid'
            else:
                act='softmax'
        if act:
            x=get_activation(act,**act_config)(x)
        return x
    return _block


def conv(
        filters,
        kernel_size=3,
        padding='same',
        seperable=False,
        batch_norm=True,
        act=True,
        act_config={},
        **conv_config):
    """conv-bn-relu"""
    def _block(x):
        if seperable:
            _conv=layers.SeparableConv2D
        else:
            _conv=layers.Conv2D
        x=_conv(
            filters=filters,
            kernel_size=kernel_size,
            padding=padding,
            **conv_config)(x)
        if batch_norm:
            x=layers.BatchNormalization()(x)
        if act:
            x=get_activation(act,**act_config)(x)
        return x
    return _block





def sepres(name,filters,filters_in=None,dilation_rate=1):
    if not filters_in:
        filters_in=filters
    def _block(x):
        res=conv(name=f'{name}-res',seperable=True,filters=filters,kernel_size=1,strides=2,dilation_rate=1)(x)
        x=conv(name=f'{name}-b1',seperable=True,filters=filters_in,kernel_size=3,strides=1,dilation_rate=1)(x)
        x=conv(name=f'{name}-b2',seperable=True,filters=filters,kernel_size=3,strides=1,dilation_rate=1)(x)
        x=conv(name=f'{name}-b3',seperable=True,filters=filters,kernel_size=3,strides=2,dilation_rate=dilation_rate)(x)
        return layers.add([res, x])
    return _block