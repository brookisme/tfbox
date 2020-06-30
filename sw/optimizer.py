import tensorflow as tf
# import tensorflow_addons as tfa


#
# CONSTANTS
#
DEFAULT_OPTIMIZER='adam'
OPTIMIZERS={
    # 'radam': tfa.optimizers.RectifiedAdam
}



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



""" TFA-RADAM:

tfa.optimizers.RectifiedAdam(
    learning_rate: Union[FloatTensorLike, Callable] = 0.001,
    beta_1: tfa.image.filters.FloatTensorLike = 0.9,
    beta_2: tfa.image.filters.FloatTensorLike = 0.999,
    epsilon: tfa.image.filters.FloatTensorLike = 1e-07,
    weight_decay: tfa.image.filters.FloatTensorLike = 0.0,
    amsgrad: bool = False,
    sma_threshold: tfa.image.filters.FloatTensorLike = 5.0,
    total_steps: Union[int, float] = 0,
    warmup_proportion: tfa.image.filters.FloatTensorLike = 0.1,
    min_lr: tfa.image.filters.FloatTensorLike = 0.0,
    name: str = 'RectifiedAdam',
    **kwargs
)

"""