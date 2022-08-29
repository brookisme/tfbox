import math
import matplotlib.pyplot as plt
import matplotlib.colors as mplib_colors
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import io
from . import helpers as h
#
# HELPERS
#
INPUT_BANDS=[0]
DESCRIPTION_HEAD="""
    {}
    * batch_index: {}
    * image_index: {}
"""
DESCRIPTION_HIST="""{}
    * {}: 
    {}
"""
class SegmentationImageWriter(object):
    
    def __init__(self,
            data_dir,
            loader,
            vmax,
            model=None,
            input_bands=INPUT_BANDS,
            target_colors=None,
            vmin=0,
            ax_h=4,
            ax_w=None,
            ax_delta=0.2,
            preserve_epoch=None,
            label_property=None,
            multioutput_rep_index=False):
        if isinstance(vmax,list):
            vmax=vmax[multioutput_rep_index or -1]
        if not target_colors:
            target_colors=h.COLORS[:vmax]
        self.input_bands=input_bands
        self.cmap=mplib_colors.ListedColormap(target_colors)
        self.vmin=vmin
        self.vmax=vmax
        self.ax_h=ax_h
        if not ax_w:
            ax_w=ax_h*(1+ax_delta)
        self.ax_w=ax_w
        self.preserve_epoch=preserve_epoch
        self.file_writer=tf.summary.create_file_writer(data_dir)
        self.loader=loader
        self.label_property=label_property
        self.model=model
        self.multioutput_rep_index=multioutput_rep_index
        
        
    def write_batch(self,batch_index,epoch=None,model=True):
        if model is True:
            model=self.model
        data=self.loader[batch_index]
        inpts,targs=data[0],data[1]
        if (self.label_property):
            labels=[r[self.label_property] for r in self.loader.batch_rows]
        else:
            labels=[None]*len(inpts)
        if model:
            preds=model(inpts)
            if isinstance(targs,list): 
                if (self.multioutput_rep_index is not None):
                    preds=[preds[self.multioutput_rep_index]]
                    targs=[targs[self.multioutput_rep_index]]
            else:
                preds=[preds]
                targs=[targs]               
            has_out_index=len(targs)>1
            for o,(p,t) in enumerate(zip(preds,targs)):
                if has_out_index:
                    out_index=o
                else:
                    out_index=False
                p=tf.argmax(p,axis=-1).numpy()
                self._save_inputs_targets_predictions(
                    out_index,
                    batch_index,
                    inpts,
                    t,p,
                    epoch,
                    labels)
        else:
            raise NotImplementedError('SegmentationImageWriter._save_inputs_targets')

        
    def _save_inputs_targets_predictions(
            self,
            out_index,
            batch_index,
            inpts,
            targs,
            preds,
            epoch,
            labels):
        for i,(inpt,targ,pred,label) in enumerate(zip(inpts,targs,preds,labels)):
            inpt,targ=self._process_input_target(inpt,targ)
            pred=self._process_prediction(pred)
            figim=self._get_figure_image(inpt,targ,pred)
            targ_hist=self._get_hist(targ)
            pred_hist=self._get_hist(pred)
            self._save_figue_image(
                out_index,
                batch_index,
                i,
                figim,
                epoch=epoch,
                label=label,
                target_hist=targ_hist,
                prediction_hist=pred_hist)
            
            
    def _process_input_target(self,inpt,targ):
        targ=np.argmax(targ,axis=-1).astype(np.uint8)
        inpt=inpt[:,:,self.input_bands]
        if inpt.shape[-1]==1:
            inpt=inpt[:,:,0]
        else:
            inpt=inpt[:,:,:3]
        return inpt, targ
    
    
    def _process_prediction(self,pred):
        return pred.astype(np.uint8)
    

    def _get_hist(self,cat_im):
        values,counts=np.unique(cat_im,return_counts=True)
        hist={ v: c for v,c in zip(values,counts) }
        return { v: hist.get(v,0) for v in range(self.vmin,self.vmax+1) }

    
    def _get_figure_image(self,inpt,targ,pred=None):
        if pred is None:
            nb_cols=2
        else:
            nb_cols=3
        figsize=(int(math.ceil(self.ax_w*nb_cols)),self.ax_h)
        fig,axs=plt.subplots(1,nb_cols,figsize=figsize)
        _=axs[0].imshow(inpt)
        _=axs[1].imshow(targ,vmin=self.vmin,vmax=self.vmax,cmap=self.cmap)
        if nb_cols==3:
            _=axs[2].imshow(pred,vmin=self.vmin,vmax=self.vmax,cmap=self.cmap)
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        image = tf.image.decode_png(buf.getvalue(), channels=3)
        image = tf.expand_dims(image, 0)
        return image

    
    def _save_figue_image(self,
            out_index,
            batch_index,
            image_index,
            image,
            epoch=None,
            label=None,
            target_hist=None,
            prediction_hist=None):
        name=f'im_{image_index}'
        if label:
            name=f'{name} - {label}'
        if out_index is not False:
            name=f'{name} [{out_index}]'
        description=DESCRIPTION_HEAD.format(label or "",batch_index,image_index)       
        if target_hist:
            description=DESCRIPTION_HIST.format(description,'target',target_hist)
        if prediction_hist:
            description=DESCRIPTION_HIST.format(description,'prediction',prediction_hist)
        with self.file_writer.as_default():
            tf.summary.image(name,image,step=epoch,description=description)









