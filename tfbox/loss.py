import tensorflow as tf
import tensorflow.keras.losses as losses
import tensorflow.keras.backend as K
#
# CONSTANTS
#
DEFAULT_LOSS='categorical_crossentropy'
DEFAULT_WEIGHTED_LOSS='weighted_categorical_crossentropy'



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



#
# CUSTOM LOSS FUNCTIIONS
#
def weighted_categorical_crossentropy(weights=None,**kwargs):
    """ weighted_categorical_crossentropy
        Args:
            * weights<ktensor|nparray|list>: crossentropy weights
        Returns:
            * weighted categorical crossentropy function
    """
    cce=losses.CategoricalCrossentropy(**kwargs)
    if weights is None:
        print('WARNING: WCCE called without weights. Defaulting to CCE')
        return cce
    else:
        print('WCCE:',weights,kwargs)
        if isinstance(weights,list) or isinstance(np.ndarray):
            weights=K.variable(weights)
        def _loss(target,output):
            unweighted_losses=cce(target,output)
            pixel_weights=tf.reduce_sum(weights*target, axis=-1)
            weighted_losses=unweighted_losses*pixel_weights
            return tf.reduce_mean(weighted_losses,axis=[1,2])
    return _loss


#
# LOSS FUNCTION DICT
#
LOSS_FUNCTIONS={
    'weighted_categorical_crossentropy': weighted_categorical_crossentropy,
    'categorical_crossentropy': losses.CategoricalCrossentropy
}
