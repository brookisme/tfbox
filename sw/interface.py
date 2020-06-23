import os
import tensorflow as tf
from sw.utils.dataloader import GroupedSeq, DATA_ROOT
from tf_toys.dataloader import FGenerator
from sw.nn.dlv3p import DLV3p
from tf_toys.model import segmentor
import sw.nn.loss
import sw.nn.optimizer
from sw.utils.tboard import TensorBoardBatchWriter
#
# CONSTANTS
#
SIZE=512
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
            nb_classes=nb_classes,
            data_root=data_root,
            batch_size=batch_size,
            cropping=cropping,
            float_cropping=float_cropping,
            size=size,
            limit=limit,
            onehot=onehot )
    return _loader


def callbacks(loader,model,directory,folder,**kwargs):
    path=os.path.join(directory,folder)
    tb=tf.keras.callbacks.TensorBoard(
        path,
        profile_batch=0,
        histogram_freq=1)
    _callbacks=[tb,_temp_image_cb(path,loader,model)]
    return _callbacks



def _temp_image_cb(data_dir,loader,model,batch_index=0):
    tbbw=TensorBoardBatchWriter(data_dir=data_dir,loader=loader,model=model)
    def lambda_func(epoch, logs):
      tbbw.write_batch(batch_index,epoch=epoch)
    return tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda_func)


def loss(loss_func,weights,**kwargs):
    return sw.nn.loss.get(loss_func,weights,**kwargs)


def optimizer(opt,**kwargs):
    return sw.nn.optimizer.get(opt,**kwargs)


def model(
        model_name=DEFAULT_MODEL_NAME,
        model_key_path=DLV3p.DEFAULT_KEY,
        size=None,
        in_ch=None,
        nb_classes=None,
        kernel_size=None,
        out_kernel_size=None,
        channels=None ):
    print('MODEL',in_ch,size)
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
            key_path=model_key_path )
    else:
        raise NotImplemented
    _input=tf.keras.Input(shape=(size,size,in_ch),name='input')
    return tf.keras.Model(_input, _model(_input))


def metrics(metrics_list=None):
    metrics_list=list(set(['accuracy']+metrics_list))
    return metrics_list

