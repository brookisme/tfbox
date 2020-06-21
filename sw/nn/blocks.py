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
            x=conv(out_ch,kernel_size=k,dilation_rate=dilation_rate)(x)
        print('conv',kernels[-1],out_ch)
        x=conv(out_ch,kernel_size=kernels[-1],dilation_rate=dilation_rate,act=False)(x)
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
        dilation_rate=1,
        strides=1,
        **conv_config):
    """conv-bn-relu"""
    if dilation_rate>1:
        strides=1
    if kernel_size in [1,(1,1)]:
        dilation_rate=1
    print('DR',filters,dilation_rate,strides)
    def _block(x,strides=strides,dilation_rate=dilation_rate):
        if seperable:
            _conv=layers.SeparableConv2D
        else:
            _conv=layers.Conv2D
        # print('DR',conv_config.pop('dilation_rate','---'))
        print('DR2',filters,dilation_rate)
        x=_conv(
            filters=filters,
            kernel_size=kernel_size,
            padding=padding,
            dilation_rate=dilation_rate,
            strides=strides,
            **conv_config)(x)
        if batch_norm:
            x=layers.BatchNormalization()(x)
        if act:
            x=get_activation(act,**act_config)(x)
        return x
    return _block





def sepres(name,filters,filters_in=None,dilation_rate=1,strides=2):
    if not filters_in:
        filters_in=filters
    print('SEPRES',filters_in,dilation_rate)
    def _block(x):
        print('SEPRES1',filters_in,dilation_rate)

        res=conv(
            name=f'{name}-res',
            seperable=True,
            filters=filters,
            kernel_size=1,
            strides=strides,
            dilation_rate=dilation_rate)(x)
        x=conv(
            name=f'{name}-b1',
            seperable=True,
            filters=filters_in,
            kernel_size=3,
            strides=1,
            dilation_rate=dilation_rate)(x)
        x=conv(
            name=f'{name}-b2',
            seperable=True,
            filters=filters,
            kernel_size=3,
            strides=1,
            dilation_rate=dilation_rate)(x)
        x=conv(
            name=f'{name}-b3',
            seperable=True,
            filters=filters,
            kernel_size=3,
            strides=strides,
            dilation_rate=dilation_rate)(x)
        return layers.add([res, x])
    return _block