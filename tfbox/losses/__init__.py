import tensorflow.keras.losses as losses
from .weighted import weighted_categorical_crossentropy, focal_cross_entropy, masked_binary_cross_entropy
#
# CONSTANTS
#
DEFAULT_LOSS='categorical_crossentropy'
DEFAULT_WEIGHTED_LOSS='weighted_categorical_crossentropy'


#
# LOSS FUNCTION DICT
#
LOSS_FUNCTIONS={
    'focal_cross_entropy': focal_cross_entropy,
    'masked_binary_cross_entropy': masked_binary_cross_entropy,
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
        else:
            loss_func=DEFAULT_LOSS
    if weights:
        kwargs['weights']=weights
    if isinstance(loss_func,str):
        loss_func=LOSS_FUNCTIONS.get(loss_func,loss_func)
    if not isinstance(loss_func,str):
        loss_func=loss_func(**kwargs)
    return loss_func


