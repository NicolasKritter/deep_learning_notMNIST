# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 17:15:20 2018

@author: nicolas
"""
import numpy as np
from six.moves import cPickle as pickle
import imageio
import os

ALL_DATA_PATH  ='notMNIST.pickle' #path of outpubFile

IMAGE_SIZE = 28 # Pixel width and height.
PIXEL_DEPTH = 255.0 # Number of levels per pixel.

#---------------Pickling the data-----------------------------------
#pickle = convert object into a byte stream, permet de stocker les matrices de données de chaque image

#normalise les  pixels des images et les charges dans une matrice
def load_letter(folder,min_num_images):
    """load the data for a single letter label"""
    # convert the entire dataset into a 3D array (image index, x, y) 
    image_files = os.listdir(folder)
    dataset = np.ndarray(shape=(len(image_files),IMAGE_SIZE,IMAGE_SIZE),dtype=np.float32)
    print("Folder: "+folder)
    num_images = 0
    for image in image_files:
        image_file = os.path.join(folder,image)
        #helps debug exception
       # print(image_file)
        try:
            #normalize data from image ((pixel -128)/128 => mean: 0 and standard deviation =0.5)
            image_data = (imageio.imread(image_file).astype(float) - PIXEL_DEPTH / 2) / PIXEL_DEPTH
            if image_data.shape !=(IMAGE_SIZE,IMAGE_SIZE):
                raise Exception('Unexpected image shape: %s' %str(image_data.shape))
            dataset[num_images,:,:] = image_data
            num_images = num_images + 1
        except (IOError,ValueError) as e:
            print ('Could not read:', image_file,':',e,'-it\'s of, skipping.')
    dataset = dataset[0:num_images,:,:]
    if num_images<min_num_images:
        raise Exception('Many fewer images than expected: %d<%d' %num_images,min_num_images)
    print('Full data tensor:', dataset.shape)
    print ('Mean: ', np.mean(dataset))
    print('Standard deviation:',np.std(dataset))
    return dataset

def maybe_pickle(data_folders,min_num_images_per_class,force=False):
    dataset_names = []
    for folder in data_folders:
        set_filename = folder +'.pickle'
        dataset_names.append(set_filename)
        if os.path.exists(set_filename) and not force:
            print('%s already present - Skippink pickling.' % set_filename)
        else:
            print('Picklink %s.' % set_filename)
            dataset = load_letter(folder,min_num_images_per_class)
            try:
                with open(set_filename,'wb') as f:
                    pickle.dump(dataset,f,pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print('Unable to save data to',set_filename,':',e)
    return dataset_names

#--------------------------Merge and Prune the dataset-----------------------
def make_arrays(nb_rows,img_size):
    if nb_rows:
        dataset = np.ndarray((nb_rows,img_size,img_size),dtype=np.float32)
        labels = np.ndarray(nb_rows,dtype=np.int32)
    else:
        dataset,labels = None,None
    return dataset,labels

def merge_datasets(pickle_files,train_size,valid_size=0):
    num_classes = len(pickle_files)
    
    valid_dataset,valid_labels = make_arrays(valid_size,IMAGE_SIZE)
    train_dataset,train_labels = make_arrays(train_size,IMAGE_SIZE)
    
    #Get the size of a class to have a balanced repartition
    vsize_per_class=valid_size // num_classes
    tsize_per_class = train_size // num_classes
    
    start_v,start_t = 0,0
    
    end_v,end_t = vsize_per_class,tsize_per_class
    
    end_l=vsize_per_class+tsize_per_class
    
    for label,pickle_file in enumerate(pickle_files):
        try:
            with open(pickle_file,'rb') as f:
                letter_set = pickle.load(f)
                #shuffle the letters to have random validation/training set
                np.random.shuffle(letter_set)
                if valid_dataset is not None:
                    valid_letter = letter_set[:vsize_per_class,:,:]
                    valid_dataset[start_v:end_v,:,:] = valid_letter
                    valid_labels[start_v:end_v] = label
                    start_v+=vsize_per_class
                    end_v+=vsize_per_class
                    
                train_letter = letter_set[vsize_per_class:end_l, :, :]
                train_dataset[start_t:end_t, :, :] = train_letter
                train_labels[start_t:end_t] = label
                
                start_t += tsize_per_class
                end_t += tsize_per_class
        except Exception as e:
            print('Unable to process data from',pickle_file,':',e)
            raise
    return valid_dataset,valid_labels,train_dataset,train_labels
    
def randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation,:,:]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels

