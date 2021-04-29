from pprint import pprint
from . import load
from . import base
from . import blocks
#
# CONSTANTS
#
DEFAULT_BTYPE='stack'
INT_BTYPE='conv'
STACK='stack'

#
# Encoder: a flexible generic encoder
# 
class Encoder(base.Model):
    #
    # CONSTANTS
    #
    NAME='Encoder'
    DEFAULT_CLASSIFIER=base.Model.GLOBAL_POOLING


    def __init__(self,
            config,
            file_name=None,
            folder=load.TFBOX,
            nb_classes=None,
            group_maps=None,
            group_nb_classes=None,
            from_logits=None,
            add_classifier=None,
            return_empty_skips=False,
            name=NAME,
            named_layers=True,
            noisy=True):
        super(Encoder, self).__init__(
            name=name,
            named_layers=named_layers,
            noisy=noisy)
        config=load.config(
            config,
            file_name or Encoder.NAME,
            folder)
        if add_classifier is None:
            add_classifier=config.get('classifier',False)
        blocks_config=config['blocks_config']
        self.stacked_blocks=[blocks.build_blocks(c) for c in blocks_config]
        self.return_empty_skips=return_empty_skips
        if add_classifier and nb_classes:
            self.set_classifier(
                nb_classes,
                config.get('classifier'),
                group_maps=group_maps,
                group_nb_classes=group_nb_classes,
                folder=folder,
                from_logits=from_logits)


    def __call__(self,inputs,training=False):
        x=inputs
        skips=[]
        for sb in self.stacked_blocks:
            for block in sb:
                xin=x
                x=block(x)
                skips=self._update_skips(block,skips,x)
        if skips or self.return_empty_skips:
            return x, skips
        else:
            return self.output(x)


    #
    # INTERNAL
    #
    def _update_skips(self,block,skips,x,force_update=False):
        try:
            if force_update or (block.is_skip): 
                skips.append(x)
        except:
            pass
        return skips









