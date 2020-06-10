import random
import numpy as np
import pandas as pd
from image_kit.handler import InputTargetHandler



BATCH_SIZE=6
GROUP_COL='group_id'
INDEX_ERROR='requested batch {} of {} batches'


class GroupedSeq(tf.keras.utils.Sequence):
        

    def __init__(self,
            data,
            group_column=GROUP_COL,
            batch_size=BATCH_SIZE,
            shuffle=True,
            **handler_kwargs):
        self.batch_size=batch_size
        self.shuffle=shuffle
        self.group_column=group_column
        self._init_dataset(data)
        self.handler=InputTargetHandler(**handler_kwargs)
        

    #
    # Sequence inteface
    #
    def __len__(self):
        """ number of batches """
        return self.nb_batches
    
    
    def __getitem__(self,batch_index):
        """ return batch """
        self.select_batch(batch_index)
        inpts=np.array([self.get_input(r) for r in self.batch_rows])
        targs=np.array([self.get_target(r) for r in self.batch_rows])
        return inpts, targs
    

    def on_epoch_end(self):
        """ on-epoch-end callback """
        self.reset()

        
    #
    # PUBLIC
    #
    def select(self,index=None):
        """ select single example """
        if index is None:
            index=np.random.randint(0,len(self.idents))
        self.index=index
        self.ident=self.idents[index]
        self.matched_rows=self.data[self.data[self.group_column]==self.ident]
        self.row=self._sample_row(data=self.matched_rows)

        
    def get(self, index):
        """ select, load, return single example """
        self.select(index)
        inpt=self.get_input()
        targ=self.get_target()
        return inpt, targ   
    
    
    def select_batch(self,batch_index):
        """ select batch """
        if batch_index>=self.nb_batches:
            raise ValueError(INDEX_ERROR.format(batch_index,self.nb_batches))
        self.batch_index=batch_index
        self.start_index=self.batch_index*self.batch_size
        self.end_index=self.start_index+self.batch_size
        self.batch_idents=self.idents[self.start_index:self.end_index]
        self.batch_rows=[ self._sample_row(ident) for ident in self.batch_idents ]

    
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
    # INTERNAL
    #
    def _init_dataset(self,data):
        if isinstance(data,str):
            data=read_csv(data,converters=CONVERTERS)
        self.data=data
        self.idents=data.loc[:,self.group_column].unique().tolist()
        self.nb_batches=int(len(self.idents)//self.batch_size)
        self.reset()
        
    
    def _sample_row(self,ident=None,data=None):
        if data is None:
            data=self.data[self.data[self.group_column]==ident]
        return data.sample().iloc[0]

    
    