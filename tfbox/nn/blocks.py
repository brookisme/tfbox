import re
from pprint import pprint
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras import activations
from . import load
import tfbox.utils.helpers as h
#
# CONSTANTS
#
DEFAULT_ACTIVATION='relu'
DEFAULT_DROPOUT=False
DEFAULT_DROPOUT_RATE=0.5
DEFAULT_MAX_POOLING={
    'pool_size': 3, 
    'strides': 2, 
    'padding': "same"
}
DEFAULT_GLOBAL_POOLING='average'
UPSAMPLE_ERROR=(
    'upsample called with scale==1. '
    'Use allow_identity=True to force.'
)

#
# HELPERS
#
def get(key,activation=False):
    if activation:
        return ACTIVATIONS[h.snake(key)]
    else:
        return BLOCKS[h.snake(key)]


def layer_name(*name_path,index=None,named=True):
    if named and name_path:
        name_path=[p for p in name_path if p] 
        name='.'.join(name_path)
        if index is not None:
            name=f'{name}-{index}'
        return name


def get_activation(act,**config):
    if act:
        if act is True:
            act=DEFAULT_ACTIVATION
        if isinstance(act,str):
            act=ACTIVATIONS[act.lower()]
        act=act(**config)
    return act


def upsample(
        x,
        scale=None,
        like=None,
        shape=None,
        rescale=1,
        mode='bilinear',
        allow_resize=True,
        force_resize=False,
        allow_identity=False):
    if scale is None:
        if shape is None:
            shape=like.shape
        scale=rescale*shape[2]/x.shape[2]
    if scale==1:
        if not allow_identity:
            raise ValueError(UPSAMPLE_ERROR)
    else:
        if allow_resize and (force_resize or (scale!=int(scale))):
            h=round(x.shape[1]*scale)
            w=round(x.shape[2]*scale)
            x=tf.image.resize(x,(h,w),method=mode)
        else:
            scale=int(scale)
            x=layers.UpSampling2D(
                size=(scale,scale),
                interpolation=mode)(x)
    return x


#
# GENERAL BLOCKS 
#
class Conv(keras.Model):
    """ Conv-BatchNorm-Activation-Dropout
    """
    #
    # PUBLIC
    #
    def __init__(self,
            filters,
            kernel_size=3,
            padding='same',
            seperable=False,
            batch_norm=True,
            act=True,
            act_config={},
            dilation_rate=1,
            strides=1,
            dropout=DEFAULT_DROPOUT,
            dropout_config={},
            act_last=False,
            is_skip=False,
            name=None,
            named_layers=True,
            **conv_config):
        super(Conv, self).__init__()
        self.block_name=name or self.name
        self.named_layers=named_layers
        if seperable:
            _conv=layers.SeparableConv2D
            cname='sepconv2d'
        else:
            _conv=layers.Conv2D
            cname='conv2d'
        if dilation_rate>1:
            strides=1
        if kernel_size in [1,(1,1)]:
            dilation_rate=1
        self.conv=_conv(
            filters=filters,
            kernel_size=kernel_size,
            padding=padding,
            dilation_rate=dilation_rate,
            strides=strides,
            name=self._layer_name(cname),
            **conv_config)
        self.bn=self._batch_norm(batch_norm)
        self.act=self._activation(act,act_config)
        self.do=self._dropout(dropout,dropout_config)
        self.act_last=act_last
        self.is_skip=is_skip


    def __call__(self,x,training=False):
        x=self.conv(x)
        if self.bn: x=self.bn(x)
        if self.act_last:
            if training and self.do: x=self.do(x)
            if self.act: x=self.act(x)
        else:
            if self.act: x=self.act(x)
            if training and self.do: x=self.do(x)         
        return x


    #
    # INTERNAL
    #
    def _layer_name(self,name):
        return layer_name(self.block_name,name,named=self.named_layers)


    def _batch_norm(self,batch_norm):
        if batch_norm:
            bn=layers.BatchNormalization(name=self._layer_name('batch_norm'))
            return bn      


    def _activation(self,act,config):
        if act:
            act=get_activation(act,name=self._layer_name('activation'),**config)
        return act


    def _dropout(self,dropout,config):
        if dropout:
            if dropout is True:
                dropout=DEFAULT_DROPOUT_RATE
            dropout=layers.Dropout(dropout,name=self._layer_name('dropout'),**config)
            return dropout



class Stack(keras.Model):
    """ (Res)Stack of Conv Blocks
    """
    #
    # CONSTANTS
    #
    IDENTITY='ident'



    #
    # PUBLIC
    #
    def __init__(self,
            filters=None,
            kernel_size=3,
            depth=1,
            filters_in=None,
            filters_out=None,
            filters_list=None,
            kernel_size_list=None,
            padding='same',
            squeeze_excitation=False,
            squeeze_excitation_ratio=16,
            residual=True,
            residual_act=True,
            seperable_residual=False,
            seperable=False,
            batch_norm=True,
            act=True,
            act_config={},
            input_dilation_rate=1,
            output_dilation_rate=1,
            output_stride=1,
            dilation_rate=1,
            max_pooling=False,
            is_skip=False,
            name=None,
            named_layers=True,
            layers_name='conv',
            **conv_config):
        super(Stack, self).__init__()
        self.block_name=name or self.name
        self.named_layers=named_layers
        self.layers_name=layers_name
        if filters_list:
            depth=len(filters_list)
        elif kernel_size_list:
            depth=len(kernel_size_list)
        self.filters_list=self._filters_list(
            filters_list,
            filters,
            filters_in,
            filters_out,
            depth)
        self.kernel_size_list=self._kernel_size_list(
            kernel_size_list,
            kernel_size,
            depth)
        self._set_config(
            act,
            output_stride,
            dilation_rate,
            max_pooling,
            seperable=seperable,
            seperable_residual=seperable_residual,
            batch_norm=batch_norm,
            padding=padding,
            act_config=act_config,
            **conv_config)
        if squeeze_excitation:
            self.se=SqueezeExcitation(
                filters,
                ratio=squeeze_excitation_ratio)
        else:
            self.se=False
        self.residual=self._residual(
            residual,
            residual_act,
            self.filters_list[-1],
            output_stride)
        self.stack=self._build_stack(
            output_stride,
            input_dilation_rate,
            output_dilation_rate)
        self.is_skip=is_skip


    def __call__(self,x,training=False,**kwargs):
        if self.residual==Stack.IDENTITY:
            res=x
        elif self.residual:
            res=self.residual(x)
        for layer in self.stack:
            x=layer(x)
        if self.se:
            x=self.se(x)
        if self.residual:
            x=layers.add([res,x])
        return x



    #
    # INTERNAL
    #
    def _filters_list(
            self,
            filters_list,
            filters,
            filters_in,
            filters_out,
            depth):
        if not filters_list:
            if depth==1:
                filters_list=[filters]
            else:
                if filters_in is None:
                    filters_in=filters            
                if filters_out is None:
                    filters_out=filters
                filters_list=[filters_in]+([filters]*(depth-2))+[filters_out]
        return filters_list


    def _kernel_size_list(self,kernel_size_list,kernel_size,depth):
        if not kernel_size_list:
            kernel_size_list=[kernel_size]*depth
        return kernel_size_list


    def _set_config(self,
            act,
            output_stride,
            dilation_rate,
            max_pooling,
            seperable,
            seperable_residual,
            **shared_config):
        self.act=act
        self.max_pooling_config=self._max_pooling_config(
            output_stride,
            dilation_rate,
            max_pooling)
        self.seperable=seperable
        self.seperable_residual=seperable_residual
        shared_config['dilation_rate']=dilation_rate
        self.shared_config=shared_config


    def _max_pooling_config(self,output_stride,dilation_rate,max_pooling):
        if max_pooling and (output_stride==2) and (dilation_rate==1):
            if max_pooling is True:
                max_pooling=DEFAULT_MAX_POOLING
            return max_pooling
        else:
            return False


    def _layer_name(self,name,index=None):
        return layer_name(self.block_name,name,index=index,named=self.named_layers)


    def _residual(self,residual,residual_act,filters,output_stride):
        if residual and (residual!=Stack.IDENTITY):
            if not residual_act:
                act=False
            else:
                act=self.act
            residual=Conv(
                filters=filters,
                kernel_size=1,
                strides=output_stride,
                seperable=self.seperable_residual,
                act=act,
                name=self._layer_name('residual'),
                named_layers=self.named_layers,
                **self.shared_config)
        return residual

    
    def _build_stack(self,
            output_stride,
            input_dilation_rate,
            output_dilation_rate):
        _layers=[] 
        last_layer_index=len(self.filters_list)-1
        for i,(f,k) in enumerate(zip(self.filters_list,self.kernel_size_list)):
            cfig=self.shared_config.copy()
            strides=cfig.pop('strides',1)
            dilation_rate=cfig.pop('dilation_rate',1)
            if (i==0):
                dilation_rate=input_dilation_rate or dilation_rate
            if (i==last_layer_index) and (not self.max_pooling_config):
                strides=output_stride or strides
                dilation_rate=output_dilation_rate or dilation_rate
            _layers.append(Conv(
                filters=f,
                kernel_size=k,
                strides=strides,
                dilation_rate=dilation_rate,
                seperable=self.seperable,
                act=self.act,
                name=self._layer_name(self.layers_name,index=i),
                named_layers=self.named_layers,
                **cfig))
            if (i==last_layer_index) and self.max_pooling_config:
                _layers.append(layers.MaxPooling2D(
                    self._layer_name('max_pooling'),
                    **self.max_pooling_config))
        return _layers





class Group(keras.Model):
    """ (Res)Parallel Group of Conv Blocks
    """
    #
    # CONSTANTS
    #
    IDENTITY='ident'



    #
    # PUBLIC
    #
    def __init__(self,
            filters,
            hidden_filters=None,
            kernel_size_list=[1,3,3],
            kernel_size_out=1,
            dilation_rate_list=[1,6,12],
            global_pooling=False,
            residual=True,
            residual_act=True,
            seperable=False,
            batch_norm=True,
            act=True,
            act_config={},
            act_last=False,
            padding='same',
            out_config={},
            is_skip=False,
            name=None,
            named_layers=True,
            layers_name='conv',
            **conv_config):
        super(Group, self).__init__()
        if padding is not 'same':
            raise NotImplementedError('currently only accepts padding=same')
        self.block_name=name or self.name
        self.named_layers=named_layers
        self.layers_name=layers_name
        self._set_config(
                kernel_size_list,
                dilation_rate_list,
                filters,
                hidden_filters,
                act,
                act_config,
                padding=padding,
                **conv_config)
        self.residual=self._residual(residual,residual_act)
        self.pooling_stack=self._global_pooling(global_pooling)
        self.group=self._build_group()
        self.out_conv=self._out_conv(kernel_size_out,out_config)
        self.is_skip=is_skip


    def __call__(self,x,training=False,**kwargs):
        if self.residual==Group.IDENTITY:
            res=x
        elif self.residual:
            res=self.residual(x)
        else:
            res=False
        x=self._call_group(x)
        if self.out_conv:
            x=self.out_conv(x)
        if res is not False:
            x=layers.add([res,x])
        return x



    #
    # INTERNAL
    #
    def _layer_name(self,name,index=None):
        return layer_name(self.block_name,name,index=index,named=self.named_layers)


    def _call_group(self,x):
        _x=[]
        for layer in self.group:
            _x.append(layer(x))
        if self.pooling_stack:
            shape=x.shape
            for layer in self.pooling_stack:
                x=layer(x)
            x=upsample(x,shape=shape)
            _x.append(x)
        return layers.Concatenate()(_x) 


    def _set_config(self,
            kernel_size_list,
            dilation_rate_list,
            filters,
            hidden_filters,
            act,
            act_config,
            **shared_config):
        self.kernel_size_list=kernel_size_list
        self.dilation_rate_list=dilation_rate_list
        self.filters=filters
        if not hidden_filters:
            hidden_filters=filters
        self.hidden_filters=hidden_filters
        self.act=act
        self.act_config=act_config
        self.shared_config=shared_config


    def _global_pooling(self,global_pooling):
        if not global_pooling:
            return False
        else:
            if global_pooling is True:
                global_pooling=DEFAULT_GLOBAL_POOLING
            _layers=[]
            if global_pooling in ['max','gmp','global_max_pooling']:
                _layers.append(layers.GlobalMaxPooling2D())
            else:
                _layers.append(layers.GlobalAveragePooling2D())
            _layers.append(layers.Lambda(
                lambda x:  tf.expand_dims(tf.expand_dims(x, axis=1),axis=1)
            ))
            _layers.append(layers.Conv2D(
                filters=self.hidden_filters,
                kernel_size=1, 
                strides=1,
                use_bias=False,
                name=self._layer_name('global_pooling'),
            ))
            return _layers


    def _residual(self,residual,residual_act):
        if residual and (residual!=Stack.IDENTITY):
            if not residual_act:
                act=False
            else:
                act=self.act
            residual=Conv(
                filters=self.hidden_filters,
                kernel_size=1,
                act=act,
                act_config=self.act_config,
                name=self._layer_name('residual',index=i),
                named_layers=self.named_layers,
                **self.shared_config)
        return residual

    
    def _build_group(self):
        _layers=[] 
        for i,(k,d) in enumerate(zip(self.kernel_size_list,self.dilation_rate_list)):
            _layers.append(Conv(
                filters=self.hidden_filters,
                kernel_size=k,
                dilation_rate=d,
                act=self.act,
                act_config=self.act_config,
                name=self._layer_name(self.layers_name,index=i),
                named_layers=self.named_layers,
                **self.shared_config))
        return _layers       


    def _out_conv(self,kernel_size,out_config):
        if self.filters:
            config=self.shared_config.copy()
            config['act']=self.act
            config['act_config']=self.act_config
            config.update(out_config)
            return Conv(
                filters=self.filters,
                kernel_size=kernel_size,
                name=self._layer_name('out_conv'),
                named_layers=self.named_layers,
                **config)




#
# BLOCKS 
#
class SqueezeExcitation(keras.Model):
    """ SqueezeExcitation
    """


    def __init__(self,filters,ratio=16,global_pooling='avg'):
        super(SqueezeExcitation, self).__init__()
        if global_pooling in ['max','gmp','global_max_pooling']:
            self.gp=layers.GlobalMaxPooling2D()
        else:
            self.gp=layers.GlobalAveragePooling2D()
        self.reduce=layers.Dense(filters//ratio,activation='relu')
        self.expand=layers.Dense(filters,activation='sigmoid')


    def __call__(self,x,training=False):
        y=self.gp(x)
        y=self.reduce(x)
        y=self.expand(x)
        return layers.multiply([x,y])




class Residual(keras.Model):
    """ Residual
    """
    IDENTITY='identity'


    def __init__(self,
            block,
            identity=False,
            filters=None,
            residual=True,
            **config):
        self.block=block
        self.residual=self._residual(
            residual,
            identity,
            filters,
            config)


    def __call__(self,x,training=False):
        if self.residual==Residual.IDENTITY:
            res=x
        elif self.residual:
            res=self.residual(x)
        else:
            res=False
        if res:
            x=layers.add([res,self.block(x)])
        return x


    #
    # INTERNAL
    #
    def _residual(self,residual,identity,filters,config):
        if residual:
            if identity:
                residual=Residual.IDENTITY
            else:
                residual=Conv(
                    filters=filters,
                    kernel_size=1,
                    **config)
        return residual



#
# PRE-CONFIGURED BLOCKS
#
class ASPP(Group):
    """ ASPP (convenience wrapper for Group)
    """
    #
    # CONSTANTS
    #
    IDENTITY='ident'



    #
    # PUBLIC
    #
    def __init__(self,
            cfig_key_path='aspp',
            cfig='blocks',
            **kwargs):
        config=load.config(cfig=cfig,key_path=cfig_key_path)
        config.update(kwargs)
        super(ASPP, self).__init__(**config)




class SegmentClassifier(keras.Model):
    """
    """
    #
    # CONSTANTS
    #
    AUTO='auto'



    #
    # STATIC
    #
    @staticmethod
    def from_config(
            cfig='blocks',
            key_path='segment_k3',
            cfig_dir=load.TFBOX,
            is_file_path=False,
            config_string=None,
            **kwargs):
        config=load.config(
            cfig=cfig,
            key_path=key_path,
            cfig_dir=cfig_dir,
            is_file_path=is_file_path,
            config_string=config_string,
            **kwargs)
        print('SegmentClassifier:')
        pprint(config)
        return SegmentClassifier(**config)


    #
    # PUBLIC
    #
    def __init__(self,
            nb_classes,
            depth=1,
            filters=None,
            filters_list=None,
            kernel_size=3,
            kernel_size_list=None,
            output_batch_norm=True,
            output_act=True,
            output_act_config={},
            seperable_preclassification=False,
            residual_preclassification=False,
            name=None,
            named_layers=True,
            **stack_config):
        super(SegmentClassifier, self).__init__()
        self.block_name=name or self.name
        self.named_layers=named_layers
        kernel_size_list=self._kernel_size_list(
            kernel_size_list,
            kernel_size,
            depth)
        filters_list=self._filters_list(
            filters_list,
            filters,
            nb_classes,
            len(kernel_size_list))
        if len(filters_list)>1:
            self.preclassifier=Stack(
                filters_list=filters_list[:-1],
                kernel_size_list=kernel_size_list[:-1],
                seperable=seperable_preclassification,
                residual=residual_preclassification,
                name=self._layer_name('stack'),
                named_layers=self.named_layers,
                **stack_config)
        else:
            self.preclassifier=False
        act=self._activation(nb_classes,output_act)
        print('OUTPUT_ACTIVATION:',act,output_act)
        self.classifier=Conv(
                filters=filters_list[-1],
                kernel_size=kernel_size_list[-1],
                batch_norm=output_batch_norm,
                act=act,
                act_config=output_act_config,
                name=self.block_name,
                named_layers=self.named_layers)



    def __call__(self,x,training=False,**kwargs):
        if self.preclassifier:
            x=self.preclassifier(x)
        return self.classifier(x)



    #
    # INTERNAL
    #
    def _layer_name(self,name,index=None):
        return layer_name(self.block_name,name,index=index,named=self.named_layers)


    def _filters_list(self,filters_list,filters,nb_classes,depth):
        if filters_list:
            if filters_list[-1]!=nb_classes:
                raise ValueError('last filters value must equal nb_classes')
        else:
            if filters is None:
                filters=nb_classes
            filters_list=[filters]*(depth-1)+[nb_classes]
        return filters_list


    def _kernel_size_list(self,kernel_size_list,kernel_size,depth):
        if not kernel_size_list:
            kernel_size_list=[kernel_size]*depth
        return kernel_size_list


    def _activation(self,nb_classes,act):
        if (act is True) or (act==SegmentClassifier.AUTO):
            if nb_classes==1:
                act='sigmoid'
            else:
                act='softmax'
        return act


#
# TBOX DICTS
#
BLOCKS={
    'conv': Conv,
    'stack': Stack,
    'group': Group,
    'squeeze_excitation': SqueezeExcitation,
    'aspp': ASPP,
    'segment_classifier': SegmentClassifier
}
ACTIVATIONS={
    'relu': layers.ReLU,
    'sigmoid': activations.sigmoid,
    'softmax': layers.Softmax
}




