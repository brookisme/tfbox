import os
import tensorflow as tf
from sw.utils.dataloader import GroupedSeq
from tf_toys.dataloader import FGenerator
from sw.nn.dlv3p import DLV3p
from tf_toys.model import segmentor
import sw.nn.loss
import sw.nn.optimizer
#
# CONSTANTS
#
SIZE=512
CROPPING=None
FLOAT_CROPPING=None
NB_TOY_BATCHES=10
DEFAULT_MODEL_NAME='dlv3p'
KERNEL_SIZE=3
OUT_KERNEL_SIZE=1
SEGMENTOR_CHANNELS=[32,64,128]








#
# PUBLIC
#
def loader(
        loader_name,
        datasets,
        batch_size,
        in_ch,
        nb_classes,
        onehot=False,
        limit=None,
        toy_size=SIZE,
        cropping=CROPPING,
        float_cropping=FLOAT_CROPPING,
        size=SIZE,
        nb_toy_batches=NB_TOY_BATCHES):
    if loader_name=='toy':
        if limit:
            nb_batches=limit*batch_size
        else:
            nb_batches=nb_toy_batches
        _loader=FGenerator(
            shape=(batch_size,toy_size,toy_size,in_ch),
            nb_cats=nb_classes-1,
            nb_batches=nb_batches,
            onehot=onehot)
    else:
        _loader=GroupedSeq(
            datasets,
            nb_categories=nb_classes,
            cropping=cropping,
            float_cropping=float_cropping,
            size=size,
            limit=limit,
            onehot=onehot )
    return _loader


def callbacks(directory,folder,**kwargs):
    tb=tf.keras.callbacks.TensorBoard(
        os.path.join(directory,folder),
        histogram_freq=1)
    _callbacks=[tb]
    return _callbacks


def loss(loss_func,weights,**kwargs):
    return sw.nn.loss.get(loss_func,weights,**kwargs)


def optimizer(opt,**kwargs):
    return sw.nn.optimizer.get(opt,**kwargs)


def model(
        model_name,
        size,
        in_ch,
        nb_classes,
        backbone,
        upsample_mode,
        kernel_size,
        out_kernel_size,
        channels ):
    print('MODEL',in_ch,size)
    # _model=Dumb(3,dilation_rate=2)
    model_name=model_name or DEFAULT_MODEL
    if model_name=='dlv3p':
        # TODO: backbone_kwargs: { } 
        _model=DLV3p(
            nb_classes=nb_classes,
            backbone=backbone,
            upsample_mode=upsample_mode )
    else:
        _model=segmentor(
            nb_classes=nb_classes,
            kernel_size=kernel_size or KERNEL_SIZE,
            out_kernel_size=out_kernel_size or OUT_KERNEL_SIZE,
            channels=SEGMENTOR_CHANNELS )
    _input=tf.keras.Input(shape=(size,size,in_ch),name='input')
    return tf.keras.Model(_input, _model(_input))


def metrics(metrics_list=None):
    metrics_list=list(set(['accuracy']+metrics_list))
    return metrics_list

