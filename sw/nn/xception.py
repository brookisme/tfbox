import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import sw.nn.blocks as blocks


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
        x=blocks.conv(filters=filters[0],strides=2,name='in_conv1')(x)
        x=blocks.conv(filters=filters[1],name='in_conv2')(x)
        return blocks.sepres('entry-sep-1',filters[2],dilation_rate=2)(x)
    return _block


def middle_flow(filters=128,block_depth=1,depth=2,dilation_rate=2):
    def _block(x):
        for b in range(depth):
            x_in=x
            for l in range(block_depth): 
                x=blocks.conv(
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
        x=blocks.sepres('exit-sepres-1',filters=filters[0],filters_in=128,dilation_rate=dilation_rate)(x)
        x=blocks.conv(name=f'exit-outconv-1',filters=filters[1],kernel_size=3,strides=1,dilation_rate=dilation_rate)(x)
        # x=blocks.conv(name=f'exit-outconv-2',filters=filters[2],kernel_size=3,strides=1,dilation_rate=dilation_rate)(x)
        # x=blocks.conv(name=f'exit-outconv-3',filters=filters[3],kernel_size=3,strides=1,dilation_rate=dilation_rate)(x)
        return x
    return _block