import tensorflow as tf
import tensorflow_addons as tfa


#
# CONSTANTS
#
DEFAULT_OPTIMIZER='radam'



#
# MAIN
#
def get(optimizer=None,**kwargs):
    if not optimizer:
        optimizer=DEFAULT_OPTIMIZER
    if isinstance(optimizer,str):
        optimizer=OPTIMIZERS.get(optimizer,optimizer)
    if not isinstance(optimizer,str):
        optimizer=optimizer(**kwargs)
    return optimizer



#
# CUSTOM OPTIMIZERS
#




#
# OPTIMIZER DICT
#
OPTIMIZERS={
    'radam': tfa.optimizers.RectifiedAdam
}