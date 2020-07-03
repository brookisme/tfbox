import tensorflow as tf
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
def weighted_categorical_crossentropy(weights=None):
    """ weighted_categorical_crossentropy
        Args:
            * weights<ktensor|nparray|list>: crossentropy weights
        Returns:
            * weighted categorical crossentropy function
    """
    if weights is None:
        return tf.keras.losses.CategoricalCrossentropy()
    else:
        if isinstance(weights,list) or isinstance(np.ndarray):
            weights=K.variable(weights)
        def _loss(target,output,from_logits=False):
            if not from_logits:
                output /= tf.reduce_sum(output,
                                        len(output.get_shape()) - 1,
                                        True)
                _epsilon = tf.convert_to_tensor(K.epsilon(), dtype=output.dtype.base_dtype)
                output = tf.clip_by_value(output, _epsilon, 1. - _epsilon)
                weighted_losses = target * tf.math.log(output) * weights
                return - tf.reduce_sum(weighted_losses,len(output.get_shape()) - 1)
            else:
                raise ValueError('WeightedCategoricalCrossentropy: not valid with logits')
    return _loss


#
# LOSS FUNCTION DICT
#
LOSS_FUNCTIONS={
    'weighted_categorical_crossentropy': weighted_categorical_crossentropy,
}
