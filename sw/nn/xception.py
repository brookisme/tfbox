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
        x=blocks.CBAD(filters=filters[0],strides=2,name='in_conv1')(x)
        x=blocks.CBAD(filters=filters[1],name='in_conv2')(x)
        x=blocks.CBADStack(
            seperable=True,
            filters=filters[2],
            dilation_rate=1,
            output_stride=2 )(x)
        return x
    return _block


def middle_flow(
        filters=128,
        flow_depth=8,
        depth=3,
        dilation_rate=1,
        residual=blocks.CBADStack.IDENTITY ):
    def _block(x):
        for _ in range(flow_depth):
            x=blocks.CBADStack(
                seperable=True,
                filters=filters,
                dilation_rate=dilation_rate,
                depth=depth,
                residual=residual )(x)
        return x
    return _block


def exit_flow(filters=[256,512,512,728],dilation_rate=2):
    def _block(x):
        x=blocks.CBADStack(
            seperable=True,
            filters=filters[0],
            dilation_rate=dilation_rate,
            output_stride=2 )(x)
        x=blocks.CBAD(
            name=f'exit-outconv-1',
            filters=filters[1],
            kernel_size=3,
            strides=1,
            dilation_rate=dilation_rate)(x)
        x=blocks.CBAD(
            name=f'exit-outconv-2',
            filters=filters[2],
            kernel_size=3,
            strides=1,
            dilation_rate=dilation_rate)(x)
        x=blocks.CBAD(
            name=f'exit-outconv-3',
            filters=filters[3],
            kernel_size=3,
            strides=1,
            dilation_rate=dilation_rate)(x)
        return x
    return _block