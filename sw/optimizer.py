import tensorflow as tf
# import tensorflow_addons as tfa
# from keras_radam import RAdam


#
# CONSTANTS
#
DEFAULT_OPTIMIZER='adam'



#
# MAIN
#
def get(optimizer=None,**kwargs):
    if not optimizer:
        optimizer=DEFAULT_OPTIMIZER
    if isinstance(optimizer,str):
        optimizer=OPTIMIZERS.get(optimizer,optimizer)
    if not isinstance(optimizer,str):
        print('optimizer!',optimizer)
        optimizer=optimizer(**kwargs)
    return optimizer



#
# CUSTOM OPTIMIZERS
#




#
# OPTIMIZER DICT
#
OPTIMIZERS={
    # 'radam': tfa.optimizers.RectifiedAdam,
    # 'radam': RAdam,
    'adam': tf.keras.optimizers.Adam
}