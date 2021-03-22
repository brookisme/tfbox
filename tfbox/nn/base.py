from pprint import pprint
import tensorflow.keras as keras
from . import load
from . import blocks
from . import addons
import tfbox.utils.helpers as h


# 
# Model: Parent class for TBox models/blocks
# 
# a simple wrapper of keras.Model with the following additions:
#
#   - an optional classifier
#   - is_skip property 
#   - standardized naming for tfbox models/blocks  
# 
class Model(keras.Model):
    #
    # CONSTANTS
    #
    NAME='TFBoxModel'
    DEFAULT_KEY=NAME
    SEGMENT='segment'
    GLOBAL_POOLING='global_pooling'
    DEFAULT_CLASSIFIER=SEGMENT


    #
    # PUBLIC
    #
    def __init__(self,
            is_skip=False,  
            name=NAME,
            named_layers=True,
            noisy=True):
        super(Model, self).__init__()
        self.classifier=None
        self.is_skip=is_skip
        self.model_name=name
        self.named_layers=named_layers


    def set_classifier(self,
            nb_classes,
            config,
            group_maps=None,
            group_nb_classes=None,
            file_name='classifier',
            folder=load.TFBOX,
            from_logits=None):
        if nb_classes:
            self.classifier=self._get_classifier(
                nb_classes,
                config,
                file_name=file_name,
                folder=folder,
                from_logits=from_logits)
        if group_maps:
            self.grouping=addons.Groups(group_maps)
            self.group_classifier=self._get_classifier(
                group_nb_classes or len(group_maps),
                config.get('group_classifier'),
                file_name=file_name,
                folder=folder,
                from_logits=from_logits)
        else:
            self.grouping=False


    def output(self,x):
        if self.classifier:
            x=self.classifier(x)
        if self.grouping:
            gx=self.grouping(x)
            if self.group_classifier:
                gx=self.group_classifier(gx)
            return [x, gx]
        else:
            return x


    def layer_name(self,group=None,index=None):
        return blocks.layer_name(self.model_name,group,index=index,named=self.named_layers)


    #
    # INTERNAL
    #
    def _get_classifier(self,
            nb_classes,
            config,
            file_name='classifier',
            folder=load.TFBOX,
            from_logits=None):
        if config:
            if from_logits in [True,False]:
                config=self._update_activation(config,from_logits)
            if nb_classes and config:
                if config is True:
                    config={}
                elif isinstance(config,str):
                    config={ 'classifier_type': config }
                else:
                    config=load.config(config,file_name,folder)
                classifier_type=config.pop( 'classifier_type', self.DEFAULT_CLASSIFIER )
                if classifier_type==Model.SEGMENT:
                    classifier=blocks.SegmentClassifier(
                        nb_classes=nb_classes,
                        **config)
                elif classifier_type==Model.GLOBAL_POOLING:
                    raise NotImplementedError('TODO: GAPClassifier')
                else:
                    raise NotImplementedError(f'{classifier_type} is not a valid classifier')
            else:
                classifier=False
            return classifier


    def _update_activation(self,config,from_logits):
        if config is True:
            config={}
        if config is False:
            config={
                'filters': False
            }
        config['output_act']=not from_logits
        return config



