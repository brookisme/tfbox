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
            config,
            file_name=None,
            folder=load.TFBOX,
            nb_classes=None,
            name=NAME,
            named_layers=True,
            noisy=True):

        super(EncoderDecoder, self).__init__(
            name=name,
            named_layers=named_layers,
            noisy=noisy)


        self.config=load.config(
            config,
            file_name or EncoderDecoder.NAME,
            folder)



        encoder_config=self.config['encoder']
        print('Encoder1:',encoder_config)
        encoder_config=load.config(
            encoder_config,
            Encoder.NAME,
            folder)
        print('Encoder2:',encoder_config)




        decoder_config=self.config['decoder']
        print('Decoder1:',decoder_config)
        decoder_config=load.config(
            self.config,
            Decoder.NAME,
            folder)
        print('Decoder2:',decoder_config)


        if nb_classes:
            self.set_classifier(
                nb_classes,
                self.config.get('classifier'),
                folder=folder)

        
        # encoder 
        self.encoder=Encoder(encoder_config,return_empty_skips=True)
        # decoder
        oc=decoder_config.get('output_conv')
        if isinstance(oc,dict):
            oc['filters']=oc.get('filters',nb_classes)
        elif oc is None:
            oc=nb_classes
        decoder_config['output_conv']=oc
        self.decoder=Decoder(decoder_config)


    def __call__(self,inputs,training=False):
        x,skips=self.encoder(inputs)
        self.decoder.set_output(inputs)
        x=self.decoder(x,skips,training)
        return self.output(x)


    #
    # INTERNAL
    #
    def _value(self,value,default):
        if value is None:
            value=default
        return value


