import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from . import xception as xcpt


BAND_AXIS=-1

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


XCEPTION='xception'
UPSAMPLE_MODE='bilinear' #tf.image.ResizeMethod.BILINEAR


def get_backbone(backbone,**kwargs):
    if isinstance(backbone,str):
        if backbone==XCEPTION:
            backbone=xcpt.model(**kwargs)
        else:
            raise NotImplemented
    return backbone

# from skimage.transform import resize
# # bottle_resized = resize(bottle, (140, 54))


def upsample(x,upsample_mode,scale=None,like=None):
    if scale is None:
        scale=int(like.shape[-2]/x.shape[-2])
    return layers.UpSampling2D(
        size=(scale,scale),
        interpolation=upsample_mode)(x)


class SegModel(tf.keras.Model):

    def __init__(self,
            out_ch,
            backbone=XCEPTION,
            upsample_mode=UPSAMPLE_MODE,
            **backbone_kwargs):
        super(SegModel, self).__init__()
        self.out_ch=out_ch
        self.upsample_mode=upsample_mode
        self.backbone=get_backbone(backbone,**backbone_kwargs)


    # def call(self, inputs, training=True):
    def __call__(self, inputs, training=False):
        x,skips=self.backbone(inputs)
        for skip in skips:
            x=self._upsample(x,like=skip)
            x=tf.concat([x,skip],axis=BAND_AXIS)
        x=self._upsample(x,like=inputs)
        x=layers.Conv2D(filters=self.out_ch,kernel_size=3,padding='same')(x)
        return layers.Softmax()(x)

    
    def _upsample(self,x,scale=None,like=None):
        if scale is None:
            scale=int(like.shape[-2]/x.shape[-2])
        return layers.UpSampling2D(
            size=(scale,scale),
            interpolation=self.upsample_mode)(x)



class DLV3p(object):


    def __init__(self,
            backbone=XCEPTION,
            upsample_mode=UPSAMPLE_MODE,
            **backbone_kwargs):
        self.backbone=get_backbone(backbone,**backbone_kwargs)


    def __call__(self,inpt):
        x,skips=self.backbone(inpt)
        for skip in skips:
            x=self._upsample(x,like=skip)
            x=tf.concat([x,skip],axis=1)
        x=self._upsample(x,like=inpt)
        return x


    def _upsample(self,x,scale=None,like=None):
        if scale is None:
            scale=int(like.shape[-2]/x.shape[-2])
        return layers.UpSampling2D(
            size=(scale,scale),
            interpolation=self.upsample_mode)(x)


#
# MODEL BLOCKS
#
def entry_flow(filters=[32,64,128]):
    def _block(x):
        x=cbr(filters=filters[0],strides=2,name='in_conv1')(x)
        x=cbr(filters=filters[1],name='in_conv2')(x)
        return sepres('entry-sep-1',filters[2],dilation_rate=2)(x)
    return _block


def middle_flow(filters=128,block_depth=3,depth=2,dilation_rate=2):
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
        x=cbr(name=f'exit-outconv-2',filters=filters[2],kernel_size=3,strides=1,dilation_rate=dilation_rate)(x)
        x=cbr(name=f'exit-outconv-3',filters=filters[3],kernel_size=3,strides=1,dilation_rate=dilation_rate)(x)
        return x
    return _block