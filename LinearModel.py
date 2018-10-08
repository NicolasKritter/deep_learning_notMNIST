# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 17:42:08 2018

@author: nicolas
"""
import numpy as np


def getScoreFromModel(train_dataset,train_labels,test_dataset,test_labels,model):
    #flatten =  transform into a 1 array ([[1],[2]]->[1,2])
    train_flatten_dataset = np.array([x.flatten() for x in train_dataset])
    test_flatten_dataset = np.array([x.flatten() for x in test_dataset])
    
    model.fit(train_flatten_dataset,train_labels)
    return model.score(test_flatten_dataset,test_labels)
    
    
def trainBySamples(all_datasets,sample_sizes,model):
    scores = []
    train_dataset = all_datasets['train_dataset']
    train_labels = all_datasets['train_labels']

    test_dataset = all_datasets['test_dataset']
    test_labels = all_datasets['test_labels']

    for sample_size in sample_sizes:
        score = getScoreFromModel(train_dataset[:sample_size],train_labels[:sample_size],test_dataset[:sample_size],test_labels[:sample_size],model)
        print("Score for "+str(sample_size)+" samples "+str(score))
        scores.append(score)
    
    return scores,sample_sizes