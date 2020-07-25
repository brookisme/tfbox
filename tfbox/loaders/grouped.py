import os
os.environ['IMAGE_BOX_BAND_ORDERING']='last'
import re
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from imagebox.handler import InputTargetHandler,BAND_ORDERING

BATCH_SIZE=6
GROUP_COL='group_id'
INDEX_ERROR='requested batch {} of {} batches'
LOCAL_DATA_ROOT=False
REMOTE_HEAD=r'^(gs|http|https)://'
INPUT_COL='input'
TARGET_COL='target'
WINDOW_INDEX_COL='window_index'
WINDOW_COL='window'
WINDOWED_GROUP_COL='__win_group_id'
INPUT_DTYPE=np.float32
TARGET_DTYPE=np.int64

class GroupedSeq(tf.keras.utils.Sequence):
        

    def __init__(self,
            data,
            nb_classes,
            batch_size=BATCH_SIZE,
            converters={},
            input_bands=None,
            augment=True,
            shuffle=True,
            limit=None,
            localize=None,
            local_data_root=LOCAL_DATA_ROOT,
            input_column=INPUT_COL,
            target_column=TARGET_COL,
            target_resolution=None,
            group_column=GROUP_COL,
            has_windows=False,
            window_index_column=WINDOW_INDEX_COL,
            window_column=WINDOW_COL,
            onehot=True,
            cropping=None,
            float_cropping=None,
            size=None,
            example_path=None,
            input_dtype=INPUT_DTYPE,
            target_dtype=TARGET_DTYPE,
            **handler_kwargs):
        self.nb_classes=nb_classes
        self.batch_size=batch_size
        self.shuffle=shuffle
        self.localize=localize
        self.onehot=onehot
        self._set_columns(
            input_column,
            target_column,
            group_column,
            has_windows,
            window_index_column,
            window_column)
        self._set_local_data_root(local_data_root)
        self._init_dataset(data,converters,limit)
        self.handler=InputTargetHandler(
            input_bands=input_bands,
            cropping=cropping,
            float_cropping=float_cropping,
            size=size,
            target_resolution=target_resolution,
            example_path=example_path,
            input_dtype=input_dtype,
            target_dtype=target_dtype,
            augment=augment,
            **handler_kwargs)


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


    def select_batch(self,batch_index):
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
            targ=to_categorical(targ,num_classes=self.nb_classes)
        return inpt, np.expand_dims(targ,-1)


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
            targs=to_categorical(targs,num_classes=self.nb_classes)
        return inpts, targs #np.expand_dims(targs,-1)

    
    def get_input(self,row=None):
        """ return input image for row or selected-row """
        if row is None:
            row=self.row
        return self.handler.input(
            row[self.input_column],
            window=self._window(row),
            return_profile=False)
    
    
    def get_target(self,row=None):
        """ return target image for row or selected-row """
        if row is None:
            row=self.row
        return self.handler.target(
            row[self.target_column],
            window=self._window(row),
            return_profile=False)

        
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
    def _set_columns(self,
            input_column,
            target_column,
            group_column,
            has_windows,
            window_index_column,
            window_column):
        self.input_column=input_column
        self.target_column=target_column
        self.has_windows=has_windows
        self.window_index_column=window_index_column
        self.window_column=window_column
        if self.has_windows:
            self.group_column=WINDOWED_GROUP_COL
            self.base_group_column=group_column
        else:
            self.group_column=group_column


    def _init_dataset(self,data,converters,limit):
        if self.has_windows:
            converters=converters or {}
            converters[self.window_column]=eval
        if isinstance(data,str):
            data=pd.read_csv(data,converters=converters)
        elif isinstance(data,list):
            if isinstance(data[0],str):
                data=[pd.read_csv(d,converters=converters) for d in data]
            data=pd.concat(data)
        if self.has_windows:
            data.loc[:,self.group_column]=data.apply(self._window_group,axis=1)
        self.idents=data.loc[:,self.group_column].unique().tolist()
        if limit:
            self.idents=self.idents[:limit*self.batch_size]
            data=data.copy()[data[self.group_column].isin(self.idents)]
        if self.localize:
            data.loc[:,self.input_column]=data[self.input_column].apply(self._localize_path)
            data.loc[:,self.target_column]=data[self.target_column].apply(self._localize_path)
        self.data=data
        self.nb_batches=int(len(self.idents)//self.batch_size)
        self.reset()
        
    
    def _window_group(self,row):
        return f'{row[self.base_group_column]}__{row[self.window_index_column]}'


    def _window(self,row):
        if self.has_windows:
            return row[self.window_column]
        else:
            return None


    def _set_local_data_root(self,local_data_root):
        if local_data_root is False:
            self.local_data_root=False
        elif isinstance(local_data_root,str):
            self.local_data_root=local_data_root
        else:
            self.local_data_root=os.getcwd() 


    def _sample_row(self,ident=None,data=None):
        if data is None:
            data=self.data[self.data[self.group_column]==ident]
        return data.sample().iloc[0]

    
    def _localize_path(self,path):
        if isinstance(self.localize,str):
            m=re.search(REMOTE_HEAD,path)
            if m:
                _,e=m.span()
                path=re.sub(f'^{self.localize}/','',path[e:])
                if self.local_data_root:
                    path=f'{self.local_data_root}/{path}'
        return path



    