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


    def __init__(self,
            nb_classes=None,
            model_config=NAME,
            key_path='xception',
            is_file_path=False,
            cfig_dir=load.TFBOX,
            classifier_config={
                'classifier_type': base.Model.GLOBAL_POOLING,
            },
            return_empty_skips=False,
            name=NAME,
            named_layers=True,
            noisy=True):
        super(Encoder, self).__init__(
            name=name,
            named_layers=named_layers,
            noisy=noisy,
            nb_classes=nb_classes,
            classifier_config=classifier_config)
        if isinstance(model_config,str):
            model_config=load.config(
                    cfig=model_config,
                    key_path=key_path,
                    is_file_path=is_file_path,
                    cfig_dir=cfig_dir,
                    noisy=noisy )
        blocks_config=model_config['blocks_config']
        self.stacked_blocks=[self._stacked_blocks(c) for c in blocks_config]
        self.return_empty_skips=return_empty_skips


    def __call__(self,inputs,training=False):
        x=inputs
        skips=[]
        for sb in self.stacked_blocks:
            for block in sb:
                xin=x
                x=block(x)
                skips=self._update_skips(block,skips,x)
        if self.classifier:
            return self.classifier(x)
        elif skips or self.return_empty_skips:
            return x, skips
        else:
            return x


    #
    # INTERNAL
    #
    def _stacked_blocks(self,config):
        if isinstance(config,int):
            btype=INT_BTYPE
        elif isinstance(config,str):
            btype=config
            config=None
        else:
            nb_keys=len(config)
            if nb_keys==1:
                btype, config=list(iter(config.items()))[0]
            else:
                btype=DEFAULT_BTYPE
        return self._get_blocks(config,btype)


    def _get_blocks(self,config,btype):
        block_stack=[]
        if isinstance(config,dict):
            layers=config.pop('layers',None)
            nb_repeats=config.pop('nb_repeats',1)
        else:
            layers=False
            nb_repeats=False
        if layers or nb_repeats:
            if not layers:
                layers=[{}]
            for i in range(nb_repeats):
                for j,c in enumerate(layers):
                    cfg=config.copy()
                    cfg.update(self._as_config_dict(c))
                    block_stack.append(
                        self._get_block(cfg,btype,index=f'{i}-{j}'))
        else:
            block_stack.append(self._get_block(config,btype))
        return block_stack


    def _get_block(self,config,btype,index=None):
        config=self._as_config_dict(config)
        name=config.get('name')
        if index and name:
            config['name']=f'{name}.{index}'
        return blocks.get(btype)(**config)


    def _update_skips(self,block,skips,x,force_update=False):
        try:
            if force_update or (block.is_skip): 
                skips.append(x)
        except:
            pass
        return skips


    def _as_config_dict(self,config):
        if not config:
            config={}
        elif isinstance(config,int):
            config={'filters': config}
        return config









