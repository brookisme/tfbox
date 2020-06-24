import random
import tensorflow.keras as keras
from sw.utils.tboard import SegmentationImageWriter, INPUT_BANDS, TARGET_COLORS



#
# CALLBACKS
#
class TBSegmentationImages(keras.callbacks.Callback):


    def __init__(self,
            data_dir,
            loader,
            input_bands=[0],
            target_colors=TARGET_COLORS,
            vmin=0,
            vmax=None,
            ax_h=4,
            ax_w=None,
            ax_delta=0.2,
            preserve_epoch=5,
            sample_batch_length=10,
            sample_batch_index=None):
        super(TBSegmentationImages,self).__init__()
        self.sample_batch_index=sample_batch_index
        self.sample_batch_indices=list(range(sample_batch_length))
        self.siw=SegmentationImageWriter(
            data_dir=data_dir,
            loader=loader,
            input_bands=input_bands,
            target_colors=target_colors,
            vmin=vmin,
            vmax=vmax,
            ax_h=ax_h,
            ax_w=ax_w,
            ax_delta=ax_delta,
            preserve_epoch=preserve_epoch)


    def on_epoch_end(self, epoch, logs={}):
        if self.sample_batch_index is None:
            self.sample_batch_index=random.choice(self.sample_batch_indices)
        self.siw.write_batch(
            self.sample_batch_index,
            epoch=epoch,
            model=self.model)


