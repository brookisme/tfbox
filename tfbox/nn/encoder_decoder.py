from . import base
from . import load
from .encoder import Encoder
from .decoder import Decoder
#
# Encoder: a flexible generic encoder
# 
class EncoderDecoder(base.Model):
    #
    # CONSTANTS
    #
    NAME='EncoderDecoder'


    def __init__(self,
            nb_classes=None,
            output_size=None,
            model_config=NAME,
            key_path=NAME,
            is_file_path=False,
            cfig_dir=load.TFBOX,
            encoder_config=None,
            encoder_key_path=None,
            encoder_is_file_path=None,
            encoder_cfig_dir=None,
            decoder_config=None,
            decoder_key_path=None,
            decoder_is_file_path=None,
            decoder_cfig_dir=None,
            classifier_position=Decoder.AFTER_UP,
            classifier_config={
                'classifier_type': base.Model.SEGMENT,
            },
            name=NAME,
            named_layers=True,
            noisy=True):
        super(EncoderDecoder, self).__init__(
            name=name,
            named_layers=named_layers,
            noisy=noisy)
        if isinstance(model_config,str):
            model_config=load.config(
                    cfig=model_config,
                    key_path=key_path,
                    is_file_path=is_file_path,
                    cfig_dir=cfig_dir,
                    noisy=noisy )
            encoder_config=model_config['encoder']
            decoder_config=model_config['decoder']
        else:
            encoder_config=load.config(
                    cfig=encoder_config,
                    key_path=self._value(encoder_key_path,key_path),
                    is_file_path=self._value(encoder_is_file_path,is_file_path),
                    cfig_dir=self._value(encoder_cfig_dir,cfig_dir),
                    noisy=noisy )
            decoder_config=load.config(
                    cfig=decoder_config,
                    key_path=self._value(decoder_key_path,key_path),
                    is_file_path=self._value(decoder_is_file_path,is_file_path),
                    cfig_dir=self._value(decoder_cfig_dir,cfig_dir),
                    noisy=noisy )
        decoder_config['classifier_config']=self._value(
                decoder_config.get('classifier_config'),
                classifier_config )
        decoder_config['classifier_position']=self._value(
            decoder_config.get('classifier_position'),
            classifier_position )
        decoder_config['output_size']=self._value(
            output_size,
            decoder_config.get('output_size'))
        self.encoder=Encoder(model_config=encoder_config,return_empty_skips=True)
        self.decoder=Decoder(nb_classes=nb_classes,model_config=decoder_config)


    def __call__(self,inputs,training=False):
        x,skips=self.encoder(inputs)
        self.decoder.set_output(inputs)
        return self.decoder(x,skips,training)


    #
    # INTERNAL
    #
    def _value(self,value,default):
        if value is None:
            value=default
        return value


