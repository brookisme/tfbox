import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


""" BLOCKS -- MOVE TO blocks.py after dev """



def cbr(filters,kernel_size=3,padding='same',seperable=False,**kwargs):
    """conv-bn-relu"""
    def _block(x):
        if seperable:
            _conv=layers.SeparableConv2D
        else:
            _conv=layers.Conv2D
        x=_conv(filters=filters,kernel_size=kernel_size,padding=padding,**kwargs)(x)
        x=layers.BatchNormalization()(x)
        return layers.ReLU()(x)
    return _block


def sepres(name,filters,filters_in=None,dilation_rate=1):
    if not filters_in:
        filters_in=filters
    def _block(x):
        res=cbr(name=f'{name}-res',seperable=True,filters=filters,kernel_size=1,strides=2,dilation_rate=1)(x)
        x=cbr(name=f'{name}-b1',seperable=True,filters=filters_in,kernel_size=3,strides=1,dilation_rate=1)(x)
        x=cbr(name=f'{name}-b2',seperable=True,filters=filters,kernel_size=3,strides=1,dilation_rate=1)(x)
        x=cbr(name=f'{name}-b3',seperable=True,filters=filters,kernel_size=3,strides=2,dilation_rate=dilation_rate)(x)
        return layers.add([res, x])
    return _block


""" ##################################### """


def model(**kwargs):
    # xception
    def _model(inpt):
        skip=entry_flow()(inpt)
        x=middle_flow()(skip)
        x=exit_flow()(x)
        return x, [skip]
    return _model


#
# MODEL BLOCKS
#
def entry_flow(filters=[32,64,128]):
    def _block(x):
        x=cbr(filters=filters[0],strides=2,name='in_conv1')(x)
        x=cbr(filters=filters[1],name='in_conv2')(x)
        return sepres('entry-sep-1',filters[2],dilation_rate=2)(x)
    return _block


def middle_flow(filters=128,block_depth=1,depth=2,dilation_rate=2):
    def _block(x):
        for b in range(depth):
            x_in=x
            for l in range(block_depth): 
                x=cbr(
                    name=f'middle-{b}-{l}',
                    seperable=True,
                    filters=filters,
                    kernel_size=3,
                    strides=1,
                    dilation_rate=dilation_rate)(x)
            x=layers.add([x_in, x])
        return x
    return _block


def exit_flow(filters=[256,512,512,728],dilation_rate=2):
    def _block(x):
        x=sepres('exit-sepres-1',filters=filters[0],filters_in=128,dilation_rate=dilation_rate)(x)
        x=cbr(name=f'exit-outconv-1',filters=filters[1],kernel_size=3,strides=1,dilation_rate=dilation_rate)(x)
        # x=cbr(name=f'exit-outconv-2',filters=filters[2],kernel_size=3,strides=1,dilation_rate=dilation_rate)(x)
        # x=cbr(name=f'exit-outconv-3',filters=filters[3],kernel_size=3,strides=1,dilation_rate=dilation_rate)(x)
        return x
    return _block