# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 16:44:03 2018

@author: nicolas
"""

#Download the dataset

"""A hook to report the progress of a download. This is mostly intended for users with
  slow internet connections. Reports every 5% change in download progress.
"""
from six.moves.urllib.request import urlretrieve
import sys
import os
import tarfile

URL = 'https://commondatastorage.googleapis.com/books1000/'
DATA_ROOT =  '.'#change to store data elsewhere
NUM_CLASSES = 10 #Number of letters

#-------------Download the Data -----------------
last_percent_reported= None
def download_progress_hook(count,blockSize,totalSize):
    global last_percent_reported
    percent = int(count*blockSize*100/totalSize)
    if last_percent_reported!=percent:
        if percent % 5 ==0:
            sys.stdout.write("%s%%" % percent)
            sys.stdout.flush()
        else:
            sys.stdout.write(".")
            sys.stdout.flush()
        last_percent_reported = percent

def maybe_download(filename,expected_bytes,force=False):
    """Download a file if not present & check the size"""
    dest_filename = os.path.join(DATA_ROOT,filename)
    if force or not os.path.exists(dest_filename):
        print('Attempting to download: ',filename)
        filename,_=urlretrieve(URL+filename,dest_filename,reporthook=download_progress_hook)
        print('\nDownload Complete!')
    statinfo = os.stat(dest_filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified',dest_filename)
    else:
        raise Exception('Failed to verify'+ dest_filename+'. Can you get it with a browser ?')
    return dest_filename

#---------Exctract the data------------------------------
#exctract files
def maybe_extract(filename,force=False):
    root = os.path.splitext(os.path.splitext(filename)[0])[0]#remove the extension
    if os.path.isdir(root) and not force:
        print('%s already present - Skipping extraction of %s.' % (root, filename))
    else:
        print('Extracting data for %s. This may take a while. Please wait.' % root)
        tar = tarfile.open(filename)
        sys.stdout.flush()
        tar.extractall(DATA_ROOT)
        tar.close()
    data_folders = [os.path.join(root,d) for d in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root,d))]
    if len(data_folders) != NUM_CLASSES:
        raise Exception('Expected %d folders, one per class. Found %d instead.' % (
        NUM_CLASSES, len(data_folders)))
    print(data_folders)
    return data_folders

