from pprint import pprint
import tensorflow.keras as keras
from . import load
import tfbox.utils.helpers as h


# 
# Model: Parent class for TBox models/blocks
# 
# a simple wrapper of keras.Model with the following additions:
#
#   - from_config() class method
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


    #
    # STATIC
    #
    @classmethod
    def from_config(
            cls,
            cfig=None,
            key_path=None,
            is_file_path=False,
            cfig_dir=load.TFBOX,
            **kwargs):
        if not cfig:
            cfig=h.snake(cls.NAME)
        if not key_path:
            key_path=cfig
        config=load.config(
            cfig=cfig,
            key_path=key_path,
            cfig_dir=cfig_dir,
            is_file_path=is_file_path,
            **kwargs)
        print(f'{cls.NAME}:')
        pprint(config)
        return cls(**config)


    def __init__(self,
            todo_classifier_args='TODO',  
            is_skip=False,  
            name=NAME,
            named_layers=True):
        super(Model, self).__init__()
        self.is_skip=is_skip
        self.model_name=name
        self.named_layers=named_layers


    def layer_name(self,group=None,index=None):
        return layer_name(self.model_name,group,index=index,named=self.named_layers)




