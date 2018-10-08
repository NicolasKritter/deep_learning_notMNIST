# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 15:36:51 2018

@author: nicolas
"""

from __future__ import print_function


import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from six.moves import cPickle as pickle

import ImportData
import DataCheckers
import PrepareData
import LinearModel


SKIP = True #skip checks
SKIP_MERGE = True # skip Merge (set False for first time)
SKIP_DUPLICATE = True #Skip duplicate search
PICKLE_FILE = os.path.join(ImportData.DATA_ROOT,PrepareData.ALL_DATA_PATH)

#Download
train_filename= ImportData.maybe_download('notMNIST_large.tar.gz', 247336696)
test_filename = ImportData.maybe_download('notMNIST_small.tar.gz', 8458043)

#Extract the dataset

train_folders = ImportData.maybe_extract(train_filename)
test_folders = ImportData.maybe_extract(test_filename)

#seed the random generator to get the same random series
np.random.seed(133)

#Problem 1: 
print("Problem 1: display samples of images to check if they looks good")
if not SKIP:
    DataCheckers.displayLettersAsImage(test_folders)

#create classes, normalize dataset & put file in manageable  format

train_dataset = PrepareData.maybe_pickle(train_folders,15000)
test_datasets =  PrepareData.maybe_pickle(test_folders,1800)

#Problem 2 display letter form the dataset array to check if it looks good
print("Problem 2 display letter form the dataset array to check if it looks good")
if not SKIP:
    DataCheckers.plotRandomLettersFromDataset(test_datasets)
    DataCheckers.plotRandomLettersFromDataset(train_dataset)
        
#Problem 3 check if the repartition is even
print ("Problem 3 check if the repartition is even")
if not SKIP:
    print("Variance test dataset: "+ str(DataCheckers.checkRepartition(test_datasets)))
    print("Variance train dataset: "+ str(DataCheckers.checkRepartition(train_dataset)))




#Merge all the pickle file to create the dataset & prune the data for a balanced repartition for each class

#Store the labels in an array from 0 t 9
#Create a validation dataset

#Change size for a slower computer
TRAIN_SIZE = 200000
VALID_SIZE = 10000
TEST_SIZE = 10000

if not SKIP_MERGE:
    valid_dataset,valid_labels,train_dataset,train_labels = PrepareData.merge_datasets(train_dataset,TRAIN_SIZE,VALID_SIZE)
    _,_,test_dataset,test_labels = PrepareData.merge_datasets(test_datasets,TEST_SIZE)

    print('Training: ', train_dataset.shape,train_labels.shape)
    print('Validation: ',valid_dataset.shape,valid_labels.shape)
    print('Testing',test_dataset.shape,test_labels.shape)

#randomize the data. It's important to have the labels well shuffled for the training and test distributions to matc

    train_dataset, train_labels = PrepareData.randomize(train_dataset, train_labels)
    test_dataset, test_labels = PrepareData.randomize(test_dataset, test_labels)
    valid_dataset, valid_labels = PrepareData.randomize(valid_dataset, valid_labels)


#save the data for later reuse:
    save = {
    'train_dataset':train_dataset,
    'train_labels':train_labels,
    'valid_dataset':valid_dataset,
    'valid_labels':valid_labels,
    'test_dataset':test_dataset,
    'test_labels':test_labels,
    }
    print('Compressed pickle size',PrepareData.savePickleData(PICKLE_FILE,save,False).st_size)

#problem 5
print("problem 5: measure overlap between training,validation & test sample")

ALL_DATASETS = pickle.load(open(PrepareData.ALL_DATA_PATH,'rb'))

if not SKIP_DUPLICATE:
    print(len(DataCheckers.findDuplicates( ALL_DATASETS['test_dataset'],ALL_DATASETS['valid_dataset'])[0]))
    print(len(DataCheckers.findDuplicates(ALL_DATASETS['valid_dataset'],ALL_DATASETS['train_dataset'])[0]))
    print(len(DataCheckers.findDuplicates( ALL_DATASETS['test_dataset'],ALL_DATASETS['train_dataset'])[0]))

#Problem 6 train a simple model with different size training samples
print("Problem 6 train a simple model with different size training samples")
#train model & get score after test

#SAMPLE_SIZE = [100,1000,5000,10000,None] #None = no limit = full dataset
SAMPLE_SIZE = [100,1000]
MODEL = LogisticRegression()
LinearModel.trainBySamples(ALL_DATASETS,SAMPLE_SIZE,MODEL)
print("Max size: "+str(len(ALL_DATASETS['train_dataset'])))