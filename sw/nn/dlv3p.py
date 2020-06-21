import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from . import xception as xcpt
from . import blocks
#
# CONSTANTS
#
BAND_AXIS=-1
DEFAULT_BACKBONE='xception'



#
# Deeplab V3+
# #
class DLV3p(tf.keras.Model):
    #
    # CONSTANTS
    #
    BACKBONES={
        'xception': xcpt.model,
        '.default': DEFAULT_BACKBONE
    }
    BILINEAR='bilinear'
    NEAREST='nearest'
    UPSAMPLE_MODE=BILINEAR



    #
    # STATIC
    #
    @staticmethod
    def get_backbone(backbone=None,**kwargs):
        if not backbone:
            backbone=DLV3p.BACKBONES.get(
                backbone,
                DLV3p.BACKBONES.get(DLV3p.BACKBONES['.default']))
        if isinstance(backbone,str):
            backbone=DLV3p.BACKBONES[backbone]
        return backbone(**kwargs)



    #
    # PUBLIC
    #
    def __init__(self,
            nb_classes,
            backbone=DEFAULT_BACKBONE,
            upsample_mode=UPSAMPLE_MODE,
            classifier_kernels=[3,1],
            classifier_act=None,
            classifier_act_config={},
            **backbone_kwargs):
        super(DLV3p, self).__init__()
        print(nb_classes,'fORCING3')
        self.upsample_mode=upsample_mode or DLV3p.UPSAMPLE_MODE
        self.backbone=DLV3p.get_backbone(backbone,**backbone_kwargs)
        self.classifier=blocks.segment_classifier(
            nb_classes,
            kernels=classifier_kernels,
            dilation_rate=2,
            act=classifier_act,
            act_config=classifier_act_config)


    def __call__(self, inputs, training=False):
        print('BBONE',inputs.shape)
        x,skips=self.backbone(inputs)
        print('UPS',x.shape,len(skips))
        for skip in skips:
            x=self._upsample(x,like=skip)
            x=tf.concat([x,skip],axis=BAND_AXIS)
        x=self._upsample(x,like=inputs)
        print('classifier',x.shape)
        x=self.classifier(x)
        print('out',x.shape)
        return x


    #
    # INTERNAL
    #
    def _upsample(self,x,scale=None,like=None):
        if scale is None:
            scale=int(like.shape[-2]/x.shape[-2])
        return layers.UpSampling2D(
            size=(scale,scale),
            interpolation=self.upsample_mode)(x)






