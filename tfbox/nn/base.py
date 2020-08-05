from pprint import pprint
import tensorflow.keras as keras
from . import load
from . import blocks
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


    def __init__(self,
            nb_classes=None,
            classifier_config={},
            classifier_key_path=None,
            classifier_is_file_path=False,
            classifier_cfig_dir=load.TFBOX,
            is_skip=False,  
            name=NAME,
            named_layers=True,
            noisy=True):
        super(Model, self).__init__()
        self.is_skip=is_skip
        self.model_name=name
        self.named_layers=named_layers
        # classifier
        if nb_classes:
            if isinstance(classifier_config,str):
                classifier_config=load.config(
                        cfig=classifier_config,
                        key_path=classifier_key_path,
                        is_file_path=classifier_is_file_path,
                        cfig_dir=classifier_cfig_dir,
                        noisy=noisy )
            classifier_type=classifier_config.pop('classifier_type')
            if classifier_type==Model.SEGMENT:
                self.classifier=blocks.SegmentClassifier(
                    nb_classes=nb_classes,
                    **classifier_config)
            elif classifier_type==Model.GLOBAL_POOLING:
                raise NotImplementedError('TODO: GAPClassifier')
            else:
                raise NotImplementedError(f'{classifier_type} is not a valid classifier')
        else:
            self.classifier=None


    def layer_name(self,group=None,index=None):
        return blocks.layer_name(self.model_name,group,index=index,named=self.named_layers)




