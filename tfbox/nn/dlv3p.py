import tensorflow as tf
from tensorflow.keras import layers
from . import xception as xcpt
from . import blocks
from . import load
#
# CONSTANTS
#
BAND_AXIS=-1



#
# Deeplab V3+
# 
class DLV3p(tf.keras.Model):
    #
    # CONSTANTS
    #
    BACKBONES={
        'xception':  { 'model': xcpt.Xception }
    }
    DEFAULT_KEY='sw'
    DEFAULTS=load.config(cfig='dlv3p',key_path=DEFAULT_KEY)
    BEFORE_UP='before'
    AFTER_UP='after'
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
        model_key=cfig=key_path=None
        if isinstance(backbone,str):
            parts=backbone.split('.')
            nb_parts=len(parts)
            model_key=parts[0]
            key_path=parts[-1]
            if nb_parts==2:
                cfig=model_key
            else:
                cfig=parts[1]
            backbone=DLV3p.BACKBONES[model_key]
        if isinstance(backbone,dict):
            model=backbone['model']
            cfig=model_key or backbone.get('config')
            key_path=key_path or backbone.get('key_path')
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
            backbone_reducer_filters=DEFAULTS.get(
                'backbone_reducer_filters',
                None),
            backbone_reducer_kernel_size=DEFAULTS.get(
                'backbone_reducer_kernel_size',
                None),
            backbone_reducer_config=DEFAULTS.get('backbone_reducer_config',{}),
            skip_reducer_filters_list=DEFAULTS.get('skip_reducer_filters_list',None),
            skip_reducer_filters=DEFAULTS.get('skip_reducer_filters',128),
            nb_skips=DEFAULTS.get('nb_skips',None),
            up_refinements_filters_list=DEFAULTS.get('up_refinements_filters_list',None),
            up_refinements_filters=DEFAULTS.get('up_refinements_filters',256),
            up_refinements_depth=DEFAULTS.get('up_refinements_depth',2),
            upsample_mode=DEFAULTS['upsample_mode'],
            classifier_position=DEFAULTS.get('classifier_position',BEFORE_UP),
            classifier_kernel_size_list=DEFAULTS['classifier_kernel_size_list'],
            classifier_filters_list=DEFAULTS.get('classifier_filters_list'),
            classifier_act=DEFAULTS.get('classifier_act',True),
            classifier_act_config=DEFAULTS.get('classifier_act_config',{})):
        super(DLV3p, self).__init__()
        self.upsample_mode=upsample_mode or DLV3p.UPSAMPLE_MODE
        self.backbone=DLV3p.build_backbone(backbone,**backbone_kwargs)
        self.nb_skips=self._nb_skips(nb_skips)
        self.classifier_position=classifier_position
        self.aspp=self._aspp(
            aspp,
            aspp_cfig_key_path,
            aspp_cfig,
            aspp_kwargs)
        self.backbone_reducer=self._get_conv(
            backbone_reducer_filters,
            backbone_reducer_kernel_size,
            backbone_reducer_config)
        self.skip_reducers=self._skip_reducers(
                skip_reducer_filters_list,
                skip_reducer_filters)
        self.up_refinements=self._up_refinements(
            up_refinements_filters_list,
            up_refinements_filters,
            up_refinements_depth)
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
        elif self.backbone_reducer:
            x=self.backbone_reducer(x)
        skips.reverse()
        for s, reducer, refines in zip(skips,self.skip_reducers,self.up_refinements):
            x=blocks.upsample(x,like=s)
            if reducer:
                s=reducer(s)
            x=tf.concat([x,s],axis=BAND_AXIS)
            for rfine in refines:
                x=rfine(x)
        if self.classifier_position==DLV3p.BEFORE_UP:
            x=self.classifier(x)
        x=blocks.upsample(x,like=inputs,interpolation=self.upsample_mode)
        if self.classifier_position==DLV3p.AFTER_UP:
            x=self.classifier(x)
        return x


    #
    # INTERNAL
    #
    def _nb_skips(self,nb_skips):
        if not nb_skips:
            try:
                nb_skips=self.backbone.stride_manager.nb_keepers
            except:
                pass
        if nb_skips:
            return nb_skips
        else:
            return 0


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


    def _skip_reducers(self,
                filters_list,
                filters):
        if not filters_list:
            if filters and self.nb_skips:
                filters_list=[filters]*self.nb_skips
        if filters_list:
            _reducers=[self._get_conv(f) for f in filters_list]
        else:
            _reducers=[False for _ in range(self.nb_skips)]
        return _reducers


    def _up_refinements(self,
                filters_list,
                filters,
                depth):
        if not filters_list:
            if filters and self.nb_skips:
                filters_list=[filters]*self.nb_skips
        if filters_list:
            _refinements=[self._get_refinements(f,depth) for f in filters_list]
        else:
            _refinements=[False for _ in range(self.nb_skips)]
        return _refinements


    def _get_refinements(self,filters,depth):
        return [ self._get_conv(filters,kernel_size=3) for _ in range(depth) ]


    def _get_conv(self,
            filters,
            kernel_size=1,
            config={}):
        if filters:
            return blocks.CBAD(
                    filters=filters,
                    kernel_size=kernel_size,
                    **(config or {}))




