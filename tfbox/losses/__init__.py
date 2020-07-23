import tensorflow.keras.losses as losses
from .weighted import weighted_categorical_crossentropy
#
# CONSTANTS
#
DEFAULT_LOSS='categorical_crossentropy'
DEFAULT_WEIGHTED_LOSS='weighted_categorical_crossentropy'


#
# LOSS FUNCTION DICT
#
LOSS_FUNCTIONS={
    'weighted_categorical_crossentropy': weighted_categorical_crossentropy,
    'categorical_crossentropy': losses.CategoricalCrossentropy
}


#
# MAIN
#
def get(loss_func=None,weights=None,**kwargs):
    if not loss_func:
        if weights:
            loss_func=DEFAULT_WEIGHTED_LOSS
            kwargs['weights']=weights
        else:
            loss_func=DEFAULT_LOSS
    if isinstance(loss_func,str):
        loss_func=LOSS_FUNCTIONS.get(loss_func,loss_func)
    if not isinstance(loss_func,str):
        loss_func=loss_func(**kwargs)
    return loss_func


