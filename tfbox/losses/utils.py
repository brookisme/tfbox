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


def scaled_loss(loss_func,scale,**kwargs):
    if isinstance(loss_func,str):
        loss_func=get(loss_func=loss_func,**kwargs)
    def _loss(target,output):
        return scale*loss_func(target,output)
    return _loss


def scaled_losses(loss_funcs,scales,kwargs_list=None,**kwargs):
    scaled_losses=[]
    if not isinstance(loss_funcs,(tuple,list)):
        loss_funcs=[loss_funcs]*len(scales)
    for i,l in enumerate(loss_funcs):
        kw=kwargs.copy()
        if kwargs_list:
            kw.update(kwargs_list[i])
        scaled_losses.append(scaled_loss(l,scales[i],**kw))
    return scaled_losses
