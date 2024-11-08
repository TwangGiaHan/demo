"""VerbalDataset"""
import os
import re
from underthesea import word_tokenize
from pathlib import Path
from sklearn.model_selection import train_test_split
from torchtext.data import Field, Example, Dataset
from tqdm.notebook import tqdm_notebook
from utils.constants import (
    ANSWER_TOKEN, ENTITY_TOKEN, SOS_TOKEN, EOS_TOKEN,
    SRC_NAME, TRG_NAME, TRAIN_PATH, TEST_PATH
)


class VerbalDataset(object):
    """VerbalDataset class"""
                                         
    def __init__(self,train,val,test):
        self.train = train
        self.val = val
        self.test = test
        self.train_data = None
        self.valid_data = None
        self.test_data = None
        self.src_field = None
        self.trg_field = None

    def _make_torchtext_dataset(self, data, fields):
        examples = [Example.fromlist(i, fields) for i in tqdm_notebook(data)]
        return Dataset(examples, fields)

    def load_data_and_fields(self, ):
        """
        Load verbalization data
        Create source and target fields
        """
        train, test, val = self.train, self.test, self.val
        
        train = train.melt(id_vars=['id',"question"],value_name="Answer")
        train = train[train['Answer'].astype(bool)].drop(['id','variable'],axis=1).values
        
        test = test.melt(id_vars=['id',"question"],value_name="Answer")
        test = test[test['Answer'].astype(bool)].drop(['id','variable'],axis=1).values
        
        val = val.melt(id_vars=['id',"question"],value_name="Answer")
        val = val[val['Answer'].astype(bool)].drop(['id','variable'],axis=1).values

        # create fields
        self.src_field = Field(tokenize=word_tokenize,
                               init_token=SOS_TOKEN,
                               eos_token=EOS_TOKEN,
                               lower=True,
                               include_lengths=True,
                               batch_first=True)
        
        self.trg_field = Field(tokenize=word_tokenize,
                               init_token=SOS_TOKEN,
                               eos_token=EOS_TOKEN,
                               lower=True,
                               batch_first=True)

        fields_tuple = [(SRC_NAME, self.src_field), (TRG_NAME, self.trg_field)]

        # create toechtext datasets
        self.train_data = self._make_torchtext_dataset(train, fields_tuple)
        self.valid_data = self._make_torchtext_dataset(val, fields_tuple)
        self.test_data = self._make_torchtext_dataset(test, fields_tuple)

        # build vocabularies
        self.src_field.build_vocab(self.train_data, min_freq=1)
        self.trg_field.build_vocab(self.train_data, min_freq=1)
        print("i am field tuple",fields_tuple)

    def get_data(self):
        """Return train, validation and test data objects"""
        return self.train_data, self.valid_data, self.test_data

    def get_fields(self):
        """Return source and target field objects"""
        return self.src_field, self.trg_field

    def get_vocabs(self):
        """Return source and target vocabularies"""
        #print('self, trg field vocab: ', self.trg_field.vocab)
        return self.src_field.vocab, self.trg_field.vocab