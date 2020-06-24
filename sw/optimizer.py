import tensorflow as tf

#
# CONSTANTS
#
DEFAULT_OPTIMIZER='adam'
OPTIMIZERS={}



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
