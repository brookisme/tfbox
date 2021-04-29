import tensorflow as tf
import abc
import warnings


#
# CONSTANTS
#
EPS=1e-8
DEFAULT_CONV_CONFIG={
    'kernel_size': 3,
    'padding': 'same'
}


#
# LAYERS
#
class Swish(tf.keras.layers.Layer):

    def __init__(self,**kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        return tf.nn.swish(inputs)




class Groups(tf.keras.layers.Layer):
    
    
    def __init__(self,group_maps):
        super(Groups, self).__init__()
        self.group_maps=group_maps
        self.nb_groups=len(group_maps)
        

    def get_config(self):
        return {"group_maps": self.group_maps}


    def call(self, x):
        grouped_list=[self._group(i,m,x) for i,m in enumerate(self.group_maps)]
        return tf.concat(grouped_list,axis=-1)


    def _group(self,index,gmap,x):
        ndim=len(x.shape)
        if ndim==4:
            if gmap==0:
                return tf.expand_dims(x[:,:,:,index]*EPS,axis=-1)
            elif gmap==1:
                return tf.expand_dims(x[:,:,:,index],axis=-1)
            else:
                indices=gmap.get('indices',[index])
                method=gmap.get('method','add')
                if method=='add':
                    return tf.keras.layers.Add()([tf.expand_dims(x[:,:,:,i],axis=-1) for i in indices])
                if method=='avg':
                    return tf.keras.layers.Average()([tf.expand_dims(x[:,:,:,i],axis=-1) for i in indices])
                else:
                    conv_cfg=DEFAULT_CONV_CONFIG.copy()
                    if isinstance(method,(int,float)):
                        conv_cfg['filters']=method 
                    else:
                        conv_cfg.update(method)
                    return tf.keras.layers.Conv2D(**conv_cfg)(tf.gather(x,indices,axis=-1))
        elif ndim==3:
            if gmap==0:
                return tf.expand_dims(x[:,:,index]*EPS,axis=-1)
            elif gmap==1:
                return tf.expand_dims(x[:,:,index],axis=-1)
            else:
                indices=gmap.get('indices',[index])
                method=gmap.get('method','add')
                if method=='add':
                    return tf.keras.layers.Add()([tf.expand_dims(x[:,:,i],axis=-1) for i in indices])
                if method=='avg':
                    return tf.keras.layers.Average()([tf.expand_dims(x[:,:,i],axis=-1) for i in indices])
                else:
                    conv_cfg=DEFAULT_CONV_CONFIG.copy()
                    if isinstance(method,(int,float)):
                        conv_cfg['filters']=method 
                    else:
                        conv_cfg.update(method)
                    return tf.keras.layers.Conv2D(**conv_cfg)(tf.gather(x,indices,axis=-1))
        else:
            raise ValueError('WTF',ndim,x.shape)






# =========================================================================================
#
#
# These are copies of TFAddon Classes, in case you are running tf2.1 (which doesn't support tfa)
#
#
# =========================================================================================

class GroupNormalization(tf.keras.layers.Layer):
    """Group normalization layer.
    copied from tfa: https://www.tensorflow.org/addons/api_docs/python/tfa/layers/GroupNormalization

    Group Normalization divides the channels into groups and computes
    within each group the mean and variance for normalization.
    Empirically, its accuracy is more stable than batch norm in a wide
    range of small batch sizes, if learning rate is adjusted linearly
    with batch sizes.
    Relation to Layer Normalization:
    If the number of groups is set to 1, then this operation becomes identical
    to Layer Normalization.
    Relation to Instance Normalization:
    If the number of groups is set to the
    input dimension (number of groups is equal
    to number of channels), then this operation becomes
    identical to Instance Normalization.
    Arguments:
        groups: Integer, the number of groups for Group Normalization.
            Can be in the range [1, N] where N is the input dimension.
            The input dimension must be divisible by the number of groups.
        axis: Integer, the axis that should be normalized.
        epsilon: Small float added to variance to avoid dividing by zero.
        center: If True, add offset of `beta` to normalized tensor.
            If False, `beta` is ignored.
        scale: If True, multiply by `gamma`.
            If False, `gamma` is not used.
        beta_initializer: Initializer for the beta weight.
        gamma_initializer: Initializer for the gamma weight.
        beta_regularizer: Optional regularizer for the beta weight.
        gamma_regularizer: Optional regularizer for the gamma weight.
        beta_constraint: Optional constraint for the beta weight.
        gamma_constraint: Optional constraint for the gamma weight.
    Input shape:
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    Output shape:
        Same shape as input.
    References:
        - [Group Normalization](https://arxiv.org/abs/1803.08494)
    """
    def __init__(
        self,
        groups=2,
        axis=-1,
        epsilon=1e-3,
        center=True,
        scale=True,
        beta_initializer="zeros",
        gamma_initializer="ones",
        beta_regularizer=None,
        gamma_regularizer=None,
        beta_constraint=None,
        gamma_constraint=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.groups = groups
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = tf.keras.initializers.get(beta_initializer)
        self.gamma_initializer = tf.keras.initializers.get(gamma_initializer)
        self.beta_regularizer = tf.keras.regularizers.get(beta_regularizer)
        self.gamma_regularizer = tf.keras.regularizers.get(gamma_regularizer)
        self.beta_constraint = tf.keras.constraints.get(beta_constraint)
        self.gamma_constraint = tf.keras.constraints.get(gamma_constraint)
        self._check_axis()


    def build(self, input_shape):

        self._check_if_input_shape_is_none(input_shape)
        self._set_number_of_groups_for_instance_norm(input_shape)
        self._check_size_of_dimensions(input_shape)
        self._create_input_spec(input_shape)

        self._add_gamma_weight(input_shape)
        self._add_beta_weight(input_shape)
        self.built = True
        super().build(input_shape)

    def call(self, inputs):

        input_shape = tf.keras.backend.int_shape(inputs)
        tensor_input_shape = tf.shape(inputs)

        reshaped_inputs, group_shape = self._reshape_into_groups(
            inputs, input_shape, tensor_input_shape
        )

        normalized_inputs = self._apply_normalization(reshaped_inputs, input_shape)

        outputs = tf.reshape(normalized_inputs, tensor_input_shape)

        return outputs

    def get_config(self):
        config = {
            "groups": self.groups,
            "axis": self.axis,
            "epsilon": self.epsilon,
            "center": self.center,
            "scale": self.scale,
            "beta_initializer": tf.keras.initializers.serialize(self.beta_initializer),
            "gamma_initializer": tf.keras.initializers.serialize(
                self.gamma_initializer
            ),
            "beta_regularizer": tf.keras.regularizers.serialize(self.beta_regularizer),
            "gamma_regularizer": tf.keras.regularizers.serialize(
                self.gamma_regularizer
            ),
            "beta_constraint": tf.keras.constraints.serialize(self.beta_constraint),
            "gamma_constraint": tf.keras.constraints.serialize(self.gamma_constraint),
        }
        base_config = super().get_config()
        return {**base_config, **config}

    def compute_output_shape(self, input_shape):
        return input_shape

    def _reshape_into_groups(self, inputs, input_shape, tensor_input_shape):

        group_shape = [tensor_input_shape[i] for i in range(len(input_shape))]
        group_shape[self.axis] = input_shape[self.axis] // self.groups
        group_shape.insert(self.axis, self.groups)
        group_shape = tf.stack(group_shape)
        reshaped_inputs = tf.reshape(inputs, group_shape)
        return reshaped_inputs, group_shape

    def _apply_normalization(self, reshaped_inputs, input_shape):

        group_shape = tf.keras.backend.int_shape(reshaped_inputs)
        group_reduction_axes = list(range(1, len(group_shape)))
        axis = -2 if self.axis == -1 else self.axis - 1
        group_reduction_axes.pop(axis)

        mean, variance = tf.nn.moments(
            reshaped_inputs, group_reduction_axes, keepdims=True
        )

        gamma, beta = self._get_reshaped_weights(input_shape)
        normalized_inputs = tf.nn.batch_normalization(
            reshaped_inputs,
            mean=mean,
            variance=variance,
            scale=gamma,
            offset=beta,
            variance_epsilon=self.epsilon,
        )
        return normalized_inputs

    def _get_reshaped_weights(self, input_shape):
        broadcast_shape = self._create_broadcast_shape(input_shape)
        gamma = None
        beta = None
        if self.scale:
            gamma = tf.reshape(self.gamma, broadcast_shape)

        if self.center:
            beta = tf.reshape(self.beta, broadcast_shape)
        return gamma, beta

    def _check_if_input_shape_is_none(self, input_shape):
        dim = input_shape[self.axis]
        if dim is None:
            raise ValueError(
                "Axis " + str(self.axis) + " of "
                "input tensor should have a defined dimension "
                "but the layer received an input with shape " + str(input_shape) + "."
            )

    def _set_number_of_groups_for_instance_norm(self, input_shape):
        dim = input_shape[self.axis]

        if self.groups == -1:
            self.groups = dim

    def _check_size_of_dimensions(self, input_shape):

        dim = input_shape[self.axis]
        if dim < self.groups:
            raise ValueError(
                "Number of groups (" + str(self.groups) + ") cannot be "
                "more than the number of channels (" + str(dim) + ")."
            )

        if dim % self.groups != 0:
            raise ValueError(
                "Number of groups (" + str(self.groups) + ") must be a "
                "multiple of the number of channels (" + str(dim) + ")."
            )

    def _check_axis(self):

        if self.axis == 0:
            raise ValueError(
                "You are trying to normalize your batch axis. Do you want to "
                "use tf.layer.batch_normalization instead"
            )

    def _create_input_spec(self, input_shape):

        dim = input_shape[self.axis]
        self.input_spec = tf.keras.layers.InputSpec(
            ndim=len(input_shape), axes={self.axis: dim}
        )

    def _add_gamma_weight(self, input_shape):

        dim = input_shape[self.axis]
        shape = (dim,)

        if self.scale:
            self.gamma = self.add_weight(
                shape=shape,
                name="gamma",
                initializer=self.gamma_initializer,
                regularizer=self.gamma_regularizer,
                constraint=self.gamma_constraint,
            )
        else:
            self.gamma = None

    def _add_beta_weight(self, input_shape):

        dim = input_shape[self.axis]
        shape = (dim,)

        if self.center:
            self.beta = self.add_weight(
                shape=shape,
                name="beta",
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                constraint=self.beta_constraint,
            )
        else:
            self.beta = None

    def _create_broadcast_shape(self, input_shape):
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis] // self.groups
        broadcast_shape.insert(self.axis, self.groups)
        return broadcast_shape




#
# BASE CLASS FOR SWA (below)
#
class AveragedOptimizerWrapper(tf.keras.optimizers.Optimizer, metaclass=abc.ABCMeta):
    """ copied from https://github.com/tensorflow/addons/blob/d26e2ed5f68092aed57016a7005ce534b1be3dce/tensorflow_addons/optimizers/average_wrapper.py#L25
    """
    def __init__(
        self, optimizer, name = "AverageOptimizer", **kwargs
    ):
        super().__init__(name, **kwargs)

        if isinstance(optimizer, str):
            optimizer = tf.keras.optimizers.get(optimizer)

        if not isinstance(optimizer, tf.keras.optimizers.Optimizer):
            raise TypeError(
                "optimizer is not an object of tf.keras.optimizers.Optimizer"
            )

        self._optimizer = optimizer
        self._track_trackable(self._optimizer, "awg_optimizer")

    def _create_slots(self, var_list):
        self._optimizer._create_slots(var_list=var_list)
        for var in var_list:
            self.add_slot(var, "average")

    def _create_hypers(self):
        self._optimizer._create_hypers()

    def _prepare(self, var_list):
        return self._optimizer._prepare(var_list=var_list)

    def apply_gradients(self, grads_and_vars, name=None, **kwargs):
        self._optimizer._iterations = self.iterations
        return super().apply_gradients(grads_and_vars, name, **kwargs)

    @abc.abstractmethod
    def average_op(self, var, average_var):
        raise NotImplementedError

    def _apply_average_op(self, train_op, var):
        average_var = self.get_slot(var, "average")
        return self.average_op(var, average_var)

    def _resource_apply_dense(self, grad, var):
        train_op = self._optimizer._resource_apply_dense(grad, var)
        average_op = self._apply_average_op(train_op, var)
        return tf.group(train_op, average_op)

    def _resource_apply_sparse(self, grad, var, indices):
        train_op = self._optimizer._resource_apply_sparse(grad, var, indices)
        average_op = self._apply_average_op(train_op, var)
        return tf.group(train_op, average_op)

    def _resource_apply_sparse_duplicate_indices(self, grad, var, indices):
        train_op = self._optimizer._resource_apply_sparse_duplicate_indices(
            grad, var, indices
        )
        average_op = self._apply_average_op(train_op, var)
        return tf.group(train_op, average_op)

    def assign_average_vars(self, var_list):
        """Assign variables in var_list with their respective averages.
        Args:
            var_list: List of model variables to be assigned to their average.
        Returns:
            assign_op: The op corresponding to the assignment operation of
            variables to their average.
        Example:
        ```python
        model = tf.Sequential([...])
        opt = tfa.optimizers.SWA(
                tf.keras.optimizers.SGD(lr=2.0), 100, 10)
        model.compile(opt, ...)
        model.fit(x, y, ...)
        # Update the weights to their mean before saving
        opt.assign_average_vars(model.variables)
        model.save('model.h5')
        ```
        """
        assign_ops = []
        for var in var_list:
            try:
                assign_ops.append(
                    var.assign(
                        self.get_slot(var, "average"), use_locking=self._use_locking
                    )
                )
            except Exception as e:
                warnings.warn("Unable to assign average slot to {} : {}".format(var, e))
        return tf.group(assign_ops)

    def get_config(self):
        config = {
            "optimizer": tf.keras.optimizers.serialize(self._optimizer),
        }
        base_config = super().get_config()
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config, custom_objects=None):
        optimizer = tf.keras.optimizers.deserialize(
            config.pop("optimizer"), custom_objects=custom_objects
        )
        return cls(optimizer, **config)

    @property
    def weights(self):
        return self._weights + self._optimizer.weights

    @property
    def lr(self):
        return self._optimizer._get_hyper("learning_rate")

    @lr.setter
    def lr(self, lr):
        self._optimizer._set_hyper("learning_rate", lr)  #

    @property
    def learning_rate(self):
        return self._optimizer._get_hyper("learning_rate")

    @learning_rate.setter
    def learning_rate(self, learning_rate):
        self._optimizer._set_hyper("learning_rate", learning_rate)



class SWA(AveragedOptimizerWrapper):
    """ copied from tfa: https://github.com/tensorflow/addons/blob/v0.12.0/tensorflow_addons/optimizers/stochastic_weight_averaging.py

    This class extends optimizers with Stochastic Weight Averaging (SWA).
    The Stochastic Weight Averaging mechanism was proposed by Pavel Izmailov
    et. al in the paper [Averaging Weights Leads to Wider Optima and
    Better Generalization](https://arxiv.org/abs/1803.05407). The optimizer
    implements averaging of multiple points along the trajectory of SGD. The
    optimizer expects an inner optimizer which will be used to apply the
    gradients to the variables and itself computes a running average of the
    variables every `k` steps (which generally corresponds to the end
    of a cycle when a cyclic learning rate is employed).
    We also allow the specification of the number of steps averaging
    should first happen after. Let's say, we want averaging to happen every `k`
    steps after the first `m` steps. After step `m` we'd take a snapshot of the
    variables and then average the weights appropriately at step `m + k`,
    `m + 2k` and so on. The assign_average_vars function can be called at the
    end of training to obtain the averaged_weights from the optimizer.
    Note: If your model has batch-normalization layers you would need to run
    the final weights through the data to compute the running mean and
    variance corresponding to the activations for each layer of the network.
    From the paper: If the DNN uses batch normalization we run one
    additional pass over the data, to compute the running mean and standard
    deviation of the activations for each layer of the network with SWA
    weights after the training is finished, since these statistics are not
    collected during training. For most deep learning libraries, such as
    PyTorch or Tensorflow, one can typically collect these statistics by
    making a forward pass over the data in training mode
    ([Averaging Weights Leads to Wider Optima and Better
    Generalization](https://arxiv.org/abs/1803.05407))
    Example of usage:
    ```python
    opt = tf.keras.optimizers.SGD(learning_rate)
    opt = tfa.optimizers.SWA(opt, start_averaging=m, average_period=k)
    ```
    """
    def __init__(
        self,
        optimizer,
        start_averaging = 0,
        average_period = 10,
        name = "SWA",
        **kwargs,
    ):
        r"""Wrap optimizer with the Stochastic Weight Averaging mechanism.
        Args:
            optimizer: The original optimizer that will be used to compute and
                apply the gradients.
            start_averaging: An integer. Threshold to start averaging using
                SWA. Averaging only occurs at `start_averaging` iters, must
                be >= 0. If start_averaging = m, the first snapshot will be
                taken after the mth application of gradients (where the first
                iteration is iteration 0).
            average_period: An integer. The synchronization period of SWA. The
                averaging occurs every average_period steps. Averaging period
                needs to be >= 1.
            name: Optional name for the operations created when applying
                gradients. Defaults to 'SWA'.
            **kwargs: keyword arguments. Allowed to be {`clipnorm`,
                `clipvalue`, `lr`, `decay`}. `clipnorm` is clip gradients by
                norm; `clipvalue` is clip gradients by value, `decay` is
                included for backward compatibility to allow time inverse
                decay of learning rate. `lr` is included for backward
                compatibility, recommended to use `learning_rate` instead.
        """
        super().__init__(optimizer, name, **kwargs)

        if average_period < 1:
            raise ValueError("average_period must be >= 1")
        if start_averaging < 0:
            raise ValueError("start_averaging must be >= 0")

        self._set_hyper("average_period", average_period)
        self._set_hyper("start_averaging", start_averaging)

    @tf.function
    def average_op(self, var, average_var):
        average_period = self._get_hyper("average_period", tf.dtypes.int64)
        start_averaging = self._get_hyper("start_averaging", tf.dtypes.int64)
        # number of times snapshots of weights have been taken (using max to
        # avoid negative values of num_snapshots).
        num_snapshots = tf.math.maximum(
            tf.cast(0, tf.int64),
            tf.math.floordiv(self.iterations - start_averaging, average_period),
        )

        # The average update should happen iff two conditions are met:
        # 1. A min number of iterations (start_averaging) have taken place.
        # 2. Iteration is one in which snapshot should be taken.
        checkpoint = start_averaging + num_snapshots * average_period
        if self.iterations >= start_averaging and self.iterations == checkpoint:
            num_snapshots = tf.cast(num_snapshots, tf.float32)
            average_value = (average_var * num_snapshots + var) / (num_snapshots + 1.0)
            return average_var.assign(average_value, use_locking=self._use_locking)

        return average_var

    def get_config(self):
        config = {
            "average_period": self._serialize_hyperparameter("average_period"),
            "start_averaging": self._serialize_hyperparameter("start_averaging"),
        }
        base_config = super().get_config()
        return {**base_config, **config}


