import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from image_kit.handler import InputTargetHandler

BATCH_SIZE=6
GROUP_COL='group_id'
INDEX_ERROR='requested batch {} of {} batches'


class GroupedSeq(tf.keras.utils.Sequence):
        

    def __init__(self,
            data,
            nb_categories,
            group_column=GROUP_COL,
            batch_size=BATCH_SIZE,
            converters=None,
            augment=True,
            shuffle=True,
            onehot=False,
            **handler_kwargs):
        self.onehot=onehot
        self.nb_categories=nb_categories
        self.batch_size=batch_size
        self.augment=augment
        self.shuffle=shuffle
        self.group_column=group_column
        self._init_dataset(data,converters)
        self.handler=InputTargetHandler(**handler_kwargs)

        
    #
    # PUBLIC
    #
    def select(self,index=None):
        """ select single example (w/o loading images) """
        if index is None:
            index=np.random.randint(0,len(self.idents))
        self.index=index
        self.ident=self.idents[index]
        self.matched_rows=self.data[self.data[self.group_column]==self.ident]
        self.row=self._sample_row(data=self.matched_rows)


    def select_batch(self,batch_index,set_augment=True,set_window=True):
        """ select batch (w/o loading images) """
        if batch_index>=self.nb_batches:
            raise ValueError(INDEX_ERROR.format(batch_index,self.nb_batches))
        self.batch_index=batch_index
        self.start_index=self.batch_index*self.batch_size
        self.end_index=self.start_index+self.batch_size
        self.batch_idents=self.idents[self.start_index:self.end_index]
        self.batch_rows=[ self._sample_row(ident) for ident in self.batch_idents ]

        
    def get(self,index,set_window=True,set_augment=True):
        """  returns single input-target pair 
        
        Args:
            - batch_index<int>: batch index
            - set_window/augment:
                if false ignore any window-cropping or augmentation
        """
        self.select(index)
        if set_window:
            self.handler.set_window()
        if set_augment:
            self.handler.set_augmentation()
        inpt=self.get_input()
        targ=self.get_target()
        if self.onehot:
            targs=to_categorical(targs,num_classes=self.nb_categories)
        return inpt, targ


    def get_batch(self,batch_index,set_window=True,set_augment=True):
        """ returns inputs-targets batch 
        
        Args:
            - batch_index<int>: batch index
            - set_window/augment:
                if false ignore any window-cropping or augmentation
                setup through `handler_kwargs`
        """
        self.select_batch(batch_index)
        if set_window:
            self.handler.set_window()
        if set_augment:
            self.handler.set_augmentation()
        inpts=np.array([self.get_input(r) for r in self.batch_rows])
        targs=np.array([self.get_target(r) for r in self.batch_rows])
        if self.onehot:
            targs=to_categorical(targs,num_classes=self.nb_categories)
        return inpts, targs

    
    def get_input(self,row=None):
        """ return input image for row or selected-row """
        if row is None:
            row=self.row
        return self.handler.input(row.s1_path,return_profile=False)
    
    
    def get_target(self,row=None):
        """ return target image for row or selected-row """
        if row is None:
            row=self.row
        return self.handler.target(row.gsw_path,return_profile=False)

        
    def reset(self):
        """ reset loader properties. (optionally) shuffle dataset """
        self.index=0
        self.ident=None
        self.matched_rows=None
        self.row=None
        self.batch_index=0
        self.start_index=None
        self.end_index=None
        self.batch_idents=None
        self.batch_rows=None
        if self.shuffle:
            random.shuffle(self.idents)


    #
    # Sequence Interface
    #
    def __len__(self):
        """ number of batches """
        return self.nb_batches
    
    
    def __getitem__(self,batch_index):
        """ return input-target batch """
        return self.get_batch(batch_index)
    

    def on_epoch_end(self):
        """ on-epoch-end callback """
        self.reset()


    #
    # INTERNAL
    #
    def _init_dataset(self,data,converters):
        if isinstance(data,str):
            data=pd.read_csv(data,converters=converters)
        self.data=data
        self.idents=data.loc[:,self.group_column].unique().tolist()
        self.nb_batches=int(len(self.idents)//self.batch_size)
        self.reset()
        
    
    def _sample_row(self,ident=None,data=None):
        if data is None:
            data=self.data[self.data[self.group_column]==ident]
        return data.sample().iloc[0]

    
    