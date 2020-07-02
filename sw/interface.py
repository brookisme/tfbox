import os
from pathlib import Path
import tensorflow as tf
from tf_toys.model import segmentor
from tf_toys.dataloader import FGenerator
from sw.utils.dataloader import GroupedSeq, DATA_ROOT
from sw.nn.dlv3p import DLV3p
import sw.loss
import sw.optimizer
import sw.callbacks
#
# CONSTANTS
#
SIZE=512
TOY_SIZE=256
CROPPING=None
FLOAT_CROPPING=None
NB_TOY_BATCHES=10
DEFAULT_MODEL_NAME='dlv3p'
DEFAULT_BACKBONE_CONFIG='small'
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
        data_root=DATA_ROOT,
        onehot=True,
        shuffle=True,
        augment=True,
        limit=None,
        toy_size=TOY_SIZE,
        cropping=CROPPING,
        float_cropping=FLOAT_CROPPING,
        size=SIZE,
        nb_toy_batches=NB_TOY_BATCHES):
    if loader_name=='fgen':
        if limit:
            nb_batches=limit*batch_size
        else:
            nb_batches=nb_toy_batches
        print('FGenerator',(batch_size,toy_size,toy_size,in_ch),onehot)
        _loader=FGenerator(
            shape=(batch_size,toy_size,toy_size,in_ch),
            nb_cats=nb_classes-1,
            nb_batches=nb_batches,
            onehot=onehot)
    else:
        _loader=GroupedSeq(
            datasets,
            nb_classes=nb_classes,
            data_root=data_root,
            batch_size=batch_size,
            cropping=cropping,
            float_cropping=float_cropping,
            size=size,
            limit=limit,
            onehot=onehot,
            shuffle=shuffle,
            augment=augment )
    return _loader


def callbacks(
        loader,
        directory,
        has_validation_data,
        name='model',
        tensorboard_folder='tensorboard',
        patience=1,
        **kwargs):
    path=os.path.join(directory,tensorboard_folder)
    model_path=os.path.join(directory,'model',name)
    model_path=f'{model_path}.best'
    Path(model_path).mkdir(parents=True, exist_ok=True)
    if has_validation_data:
        monitor='val_loss'
    else:
        monitor='loss'
    tb=tf.keras.callbacks.TensorBoard(
        path,
        profile_batch=0,
        histogram_freq=1)
    es=tf.keras.callbacks.EarlyStopping(
        monitor=monitor, 
        patience=patience, 
        restore_best_weights=False )
    mc=tf.keras.callbacks.ModelCheckpoint(
        monitor=monitor,
        filepath=model_path, 
        save_best_only=True)
    _callbacks=[
        tb,es,mc,
        sw.callbacks.TBSegmentationImages(path,loader)]
    return _callbacks


def loss(loss_func,weights,**kwargs):
    if weights:
        weights=[float(w) for w in weights]
    out= sw.loss.get(loss_func,weights,**kwargs)
    return out

def optimizer(opt,**kwargs):
    return sw.optimizer.get(opt,**kwargs)


def model(
        nb_classes,
        model_name=DEFAULT_MODEL_NAME,
        model_key_path=None,
        backbone=None,
        size=None,
        in_ch=None,
        kernel_size=None,
        out_kernel_size=None,
        channels=None ):
    model_name=model_name or DEFAULT_MODEL_NAME
    if model_name=='toy':
        _model=segmentor(
            nb_classes=nb_classes,
            kernel_size=kernel_size or KERNEL_SIZE,
            out_kernel_size=out_kernel_size or OUT_KERNEL_SIZE,
            channels=channels or SEGMENTOR_CHANNELS )
    elif model_name=='dlv3p':
        _model=DLV3p.from_config(
            nb_classes=nb_classes,
            key_path=model_key_path or DLV3p.DEFAULT_KEY,
            backbone=backbone or DLV3p.DEFAULTS['backbone'])
    else:
        raise NotImplemented
    _input=tf.keras.Input(shape=(size,size,in_ch),name='input')
    return tf.keras.Model(_input, _model(_input))


def metrics(metrics_list=None):
    metrics_list=list(set(['accuracy']+metrics_list))
    return metrics_list

