import tensorflow as tf
from tensorflow.keras import layers
from . import xception as xcpt
from . import blocks
from . import load
#
# CONSTANTS
#
BAND_AXIS=-1
BACKBONES={
    'xception':  { 'model': xcpt.Xception, 'config': 'xception' },
    'xception_small':  { 'model': xcpt.Xception, 'config': 'xception.small' },
}


#
# Deeplab V3+
# 
class DLV3p(tf.keras.Model):
    #
    # CONSTANTS
    #
    DEFAULT_KEY='sw'
    DEFAULTS=load.config(cfig='dlv3p',key_path=DEFAULT_KEY)
    #
    # STATIC
    #
    @staticmethod
    def from_config(
            key_path=DEFAULT_KEY,
            cfig='dlv3p',
            is_file_path=False,
            **kwargs):
        config=load.config(
            cfig=cfig,
            key_path=key_path,
            is_file_path=is_file_path,
            **kwargs)
        return DLV3p(**config)



    @staticmethod
    def build_backbone(backbone,**kwargs):
        if isinstance(backbone,str):
            backbone=BACKBONES[backbone]
        if isinstance(backbone,dict):
            model=backbone['model']
            cfig=backbone['config']
            key_path=backbone.get('key_path')
            is_file_path=backbone.get('is_file_path',False)
            config=load.config(
                cfig=cfig,
                key_path=key_path,
                is_file_path=is_file_path,
                **kwargs)
        else:
            model=backbone
            config=kwargs
        return model(**config)



    #
    # PUBLIC
    #
    def __init__(self,
            nb_classes,
            backbone=DEFAULTS['backbone'],
            backbone_kwargs=DEFAULTS.get('backbone_kwargs',{}),
            aspp=DEFAULTS.get('aspp',True),
            aspp_cfig_key_path=DEFAULTS.get('aspp_cfig_key_path','aspp'),
            aspp_cfig=DEFAULTS.get('aspp_cfig','blocks'),
            aspp_kwargs=DEFAULTS.get('aspp_kwargs',{}),
            upsample_mode=DEFAULTS['upsample_mode'],
            classifier_kernel_size_list=DEFAULTS['classifier_kernel_size_list'],
            classifier_filters_list=DEFAULTS.get('classifier_filters_list'),
            classifier_act=DEFAULTS.get('classifier_act'),
            classifier_act_config=DEFAULTS.get('classifier_act_config',{})):
        super(DLV3p, self).__init__()
        self.upsample_mode=upsample_mode or DLV3p.UPSAMPLE_MODE
        self.backbone=DLV3p.build_backbone(backbone,**backbone_kwargs)
        self.aspp=self._aspp(
            aspp,
            aspp_cfig_key_path,
            aspp_cfig,
            aspp_kwargs)
        self.classifier=blocks.SegmentClassifier(
            nb_classes=nb_classes,
            filters_list=classifier_filters_list,
            kernel_size_list=classifier_kernel_size_list,
            output_act=classifier_act,
            output_act_config=classifier_act_config)


    def __call__(self, inputs, training=False):
        x,skips=self.backbone(inputs)
        if self.aspp:
            x=self.aspp(x)
        for skip in skips:
            x=self._upsample(x,like=skip)
            x=tf.concat([x,skip],axis=BAND_AXIS)
        x=blocks.upsample(x,like=inputs,interpolation=self.upsample_mode)
        x=self.classifier(x)
        return x


    #
    # INTERNAL
    #
    def _aspp(self,
            aspp,
            cfig_key_path,
            cfig,
            config):
        if aspp:
            return blocks.ASPP(
                    cfig_key_path=cfig_key_path,
                    cfig=cfig,
                    **config)








