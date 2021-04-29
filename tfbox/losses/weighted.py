import tensorflow as tf
import tensorflow.keras.losses as losses
import tensorflow.keras.backend as K
from tensorflow.python.framework import ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.framework import constant_op
#
# CONSTANTS
#
EPS=1e-8
CCE_ARGS=[
    'from_logits',
    'label_smoothing',
    'reduction',
    'name'
]


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
    kwargs={ k: kwargs[k] for k in kwargs.keys() if k in CCE_ARGS }
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



def focal_cross_entropy(
        gamma=2,
        alpha=0.25,
        from_logits=False,
        from_logits_method='softmax',
        weights=None,
        ignore_labels=None,
        nb_classes=None,
        **kwargs):
    """ generalized focal loss that defaults to "softmax-focal-loss"
    """
    if (not weights) and ignore_labels:
        weights=[1]*nb_classes
        for l in ignore_labels:
            weights[l]=0
    print('FOCAL LOSS',gamma,alpha,weights)
    def _loss(target,output):
        return focalized_categorical_crossentropy(
            target,
            output,
            alpha=alpha,
            gamma=gamma,
            from_logits=from_logits,
            from_logits_method=from_logits_method,
            weights=weights)
    return _loss


def masked_binary_cross_entropy(
        mask_index=0,
        category_index=None,
        pos_weight=None,
        **kwargs):
    """ 
    """
    if category_index:
        if (mask_index!=0) and (mask_index!=-1):
            raise ValueError('if category_index index is None, mask_index must be 0 or -1') 
    if pos_weight is None:
        print('MASKED BCE',mask_index,category_index,kwargs)
        bce=losses.BinaryCrossentropy(**kwargs)
    else:
        if not kwargs.get('from_logits'):
            raise ValueError('logits required with pos_weight')
        def bce(target,output):
            print('MASKED WCCE',pos_weight)
            return tf.nn.weighted_cross_entropy_with_logits(
                target, 
                output, 
                pos_weight, 
                name=kwargs.get('name','masked_binary_cross_entropy'))
    def _loss(target,output):
        msk=target[:,:,:,mask_index]!=1
        if category_index:
            target=target[:,:,:,category_index]
        elif mask_index==-1:
            target=target[:,:,:,:-1]
        else:
            target=target[:,:,:,1:]
        target=tf.boolean_mask(target,msk)
        output=tf.boolean_mask(output,msk)
        return bce(target,output)
    return _loss



#
# METHODS
#
def pixel_weighted_categorical_crossentropy(weights,target, output, from_logits=False, axis=-1):
    """ pixel weighted version of tf.keras.backend.categorical_crossentropy

    copy of https://github.com/tensorflow/tensorflow/blob/v2.3.1/tensorflow/python/keras/backend.py#L4640-L4708
    except for last line where weights are introduced
    """
    target = ops.convert_to_tensor_v2(target)
    output = ops.convert_to_tensor_v2(output)
    target.shape.assert_is_compatible_with(output.shape)
    if from_logits:
        return nn.softmax_cross_entropy_with_logits_v2(
            labels=target, logits=output, axis=axis)
    if (not isinstance(output, (ops.EagerTensor, variables_module.Variable)) and
            output.op.type == 'Softmax') and not hasattr(output, '_keras_history'):
        assert len(output.op.inputs) == 1
        output = output.op.inputs[0]
        return nn.softmax_cross_entropy_with_logits_v2(
            labels=target, logits=output, axis=axis)
    output = output / math_ops.reduce_sum(output, axis, True)
    epsilon_ = _constant_to_tensor(epsilon(), output.dtype.base_dtype)
    output = clip_ops.clip_by_value(output, epsilon_, 1. - epsilon_)
    return -math_ops.reduce_sum(weights * target * math_ops.log(output), axis)



def focalized_categorical_crossentropy(
        target,
        output,
        alpha=0.25,
        gamma=2,
        from_logits=False,
        from_logits_method='sigmoid',
        weights=None):
    """ focal version of """
    if from_logits:
        if from_logits_method=='sigmoid':
            prob=tf.sigmoid(output)
        elif from_logits_method=='softmax':
            prob=tf.nn.softmax(output)
        else:
            prob=from_logits_method(output)
    else:
        prob=output
    p_t = (target * prob)+((1 - target)*(1 - prob))
    alpha_factor = 1.0
    modulating_factor = 1.0
    if alpha:
        alpha=tf.convert_to_tensor(alpha, dtype=K.floatx())
        alpha_factor = target * alpha + (1 - target) * (1 - alpha)
    if gamma:
        gamma=tf.convert_to_tensor(gamma, dtype=K.floatx())
        modulating_factor=tf.pow((1.0 - p_t), gamma)
    focal_weights=alpha_factor*modulating_factor
    if weights:
        focal_weights=weights*focal_weights
    return pixel_weighted_categorical_crossentropy(focal_weights,target,output,from_logits=from_logits)


#
# INTERNAL
#
def _constant_to_tensor(x, dtype):
    """ copy https://github.com/tensorflow/tensorflow/blob/v2.3.1/tensorflow/python/keras/backend.py#L805
    """
    return constant_op.constant(x, dtype=dtype)


