import tensorflow as tf
import tensorflow.keras.losses as losses
import tensorflow.keras.backend as K

EPS=1e-8

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
    if weights is None:
        print('WARNING: WCCE called without weights. Defaulting to CCE')
        return losses.CategoricalCrossentropy(**kwargs)
    else:
        if isinstance(weights,list) or isinstance(np.ndarray):
            weights=K.variable(weights)
        kwargs['reduction']=tf.keras.losses.Reduction.NONE
        print('WCCE:',weights,kwargs)
        cce=losses.CategoricalCrossentropy(**kwargs)
        def _loss(target,output):
            unweighted_losses=cce(target,output)
            pixel_weights=tf.reduce_sum(weights*target, axis=-1)
            weighted_losses=unweighted_losses*pixel_weights
            return tf.reduce_mean(weighted_losses)
    return _loss

