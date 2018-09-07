# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 16:59:50 2018

@author: nicolas
"""
import random
import os
from IPython.display import Image
import matplotlib.pyplot as plt
from six.moves import cPickle as pickle
import numpy as np
import hashlib

def displayLettersAsImage(folders_list):
    for folder in folders_list:
        letter = folder[-1]
        image = random.choice(os.listdir(folder))
        path = os.path.join(folder,image)
        #can't use iPython  display on windows (api problmen)
        img = Image.open(path)
        img.show()
        print (letter)
        
def plotRandomLettersFromDataset(dataset):
    for letter in dataset:
        letter_list = pickle.load(open(letter,"rb"))
        letter_plot = random.choice(letter_list)
        #%matplotlib notebook
        plt.figure()
        plt.title(letter)
        plt.imshow(letter_plot)
def checkRepartition(dataset):
    data = []
    for letter in dataset:
        letter_list = pickle.load(open(letter,"rb"))
        lenght = len(letter_list)
        data.append(lenght);
        print(letter + " size: "+str(lenght))
    return np.var(data)
    
def findDuplicates(dataset1,dataset2):
    if len(dataset2)>len(dataset1):
        dataset1,dataset2 = dataset2,dataset1
    hashes1 = [hashlib.sha1(x).hexdigest() for x in dataset1]
    dup_index = []
    for i in range(0,len(dataset2)):
        if hashlib.sha1(dataset2[i]).hexdigest() in hashes1:
            dup_index.append(i)
    return dup_index,dataset2