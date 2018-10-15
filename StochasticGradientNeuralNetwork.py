# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 16:09:03 2018

@author: nicolas
"""
#https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/udacity/2_fullyconnected.ipynb

from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range

PICKLE_FILE='notMNIST.pickle'

#reload the data generated in notmnist_1

with open (PICKLE_FILE,'rb') as f:
    save = pickle.load(f)
    train_dataset = save['train_dataset']
    train_labels = save['train_labels']
    valid_dataset = save['valid_dataset']
    valid_labels = save['valid_labels']
    test_dataset = save['test_dataset']
    test_labels = save['test_labels']
    del save #help gc to free memory
    print('Training set',train_dataset.shape,train_labels.shape)
    print('Validation set',valid_dataset.shape,valid_labels.shape)
    print('Test set',test_dataset.shape,test_labels.shape)

#reformat the data into a more adapted shape

IMAGE_SIZE=28
NUMBER_OF_LABELS = 10

def reformat(dataset,labels):
    dataset = dataset.reshape((-1,IMAGE_SIZE*IMAGE_SIZE)).astype(np.float32)
    # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
    labels = (np.arange(NUMBER_OF_LABELS)==labels[:,None]).astype(np.float32)
    return dataset,labels

train_dataset,train_labels = reformat(train_dataset,train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)

print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

# With gradient descent training, even this much data is prohibitive.

TRAIN_SUBSET = 10000

#How TF work: first describe the INPUTS,VARIABLES and the OPERATION as nodes in a graph
#then run the operations as many times as you want
graph = tf.Graph()

with graph.as_default():
    #INPUTS 

    #the training,validating,testing data are loaded into constant 
    #attached to the graph
    
    #Train a subset for faster turnaround
    TF_TRAIN_DATASET = tf.constant(train_dataset[:TRAIN_SUBSET,:])
    TF_TRAIN_LABELS = tf.constant(train_labels[:TRAIN_SUBSET])
    #Use all set for test & validation
    TF_VALID_DATASET = tf.constant(valid_dataset)
    TF_TEST_DATASET = tf.constant(test_dataset)
    
    #VARIABLES: Parameters used for the training
    
    #Weight matrix initialized with random values using a truncated normal distribution
    weights = tf.Variable(tf.truncated_normal([IMAGE_SIZE * IMAGE_SIZE,NUMBER_OF_LABELS]))
    
    #biases initialize at 0
    biases = tf.Variable(tf.zeros([NUMBER_OF_LABELS]))
    
    #OPERATION
    # Training computation: Inputs * Weight + biases
    logits = tf.matmul(TF_TRAIN_DATASET,weights)+ biases
    
    # loss: Comput the softmax & cross entropy and take the average for all the training set
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=TF_TRAIN_LABELS,logits=logits))
    
    #OPTIMIZER
    #Find the minimum of this loss using gradient descent
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
    
    #Predictions for the traning,validation & test data
    #(not par of the training but help report accuray as we train)
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(tf.matmul(TF_VALID_DATASET,weights)+biases)
    test_prediction = tf.nn.softmax(tf.matmul(TF_TEST_DATASET,weights)+biases)
    
#Run computation and iterate

NUM_STEPS = 801

def accuracy(predictions,labels):
    return (100.0*np.sum(np.argmax(predictions,1)==np.argmax(labels,1))/predictions.shape[0])

with tf.Session(graph=graph) as session:
    #One time operation to unsures the parameters are correctly initialized
    #(Check random weights of the matrix and 0 for biases)
    tf.global_variables_initializer().run()
    print('Initialized')
    
    
    for step in range(NUM_STEPS):
        #run the computation, get the loss and prediction in return
        _,run_loss,predictions = session.run([optimizer,loss,train_prediction])
        
        #Print the progress
        if (step % 100 ==0):
            print('Loss at step %d: %f' % (step,run_loss))
            print('Training accuracy: %.1f%%' % accuracy(
            predictions,train_labels[:TRAIN_SUBSET,:]))
            #calling eval() is like calling run() but return one numpy array
            #This recompute all its graph dependencies
            print('Validation accuracy: %.1f%%' % accuracy(
            valid_prediction.eval(),valid_labels))
    print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(),test_labels))

#STOCHASTIC GRADIENT DESCENT TRAINING
#Much faster technique, create placeholder node to feed at every run()

BATCH_SIZE = 128

graph = tf.Graph()

with graph.as_default():
    #placeholder: variables of the graph, name of the key  =  name of the variable
    #INPUT
    #Put training data in placeholvder with a training batch
    tf_train_dataset = tf.placeholder(tf.float32,shape=(BATCH_SIZE,IMAGE_SIZE*IMAGE_SIZE))
    tf_train_labels = tf.placeholder(tf.float32,shape=(BATCH_SIZE,NUMBER_OF_LABELS))
    TF_VALID_DATASET = tf.constant(valid_dataset)
    TF_TEST_DATASET = tf.constant(test_dataset)
    
    #VARIABLES: Parameters used for the training
    
    #Weight matrix initialized with random values using a truncated normal distribution
    weights = tf.Variable(tf.truncated_normal([IMAGE_SIZE * IMAGE_SIZE,NUMBER_OF_LABELS]))
    
    #biases initialize at 0
    biases = tf.Variable(tf.zeros([NUMBER_OF_LABELS]))
    
    #OPERATION
    # Training computation: Inputs * Weight + biases
    logits = tf.matmul(tf_train_dataset,weights)+ biases
    
    # loss: Comput the softmax & cross entropy and take the average for all the training set
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels,logits=logits))
    
    #OPTIMIZER
    #Find the minimum of this loss using gradient descent
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
    
    #Predictions for the traning,validation & test data
    #(not par of the training but help report accuray as we train)
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(tf.matmul(TF_VALID_DATASET,weights)+biases)
    test_prediction = tf.nn.softmax(tf.matmul(TF_TEST_DATASET,weights)+biases)

#Run the stochastic gradient descent training
NUM_STEPS = 3001

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print("Initialized")
    for step in range(NUM_STEPS):
        #Pick an offset within the randomized training data
        offset = (step*BATCH_SIZE) %(train_labels.shape[0] - BATCH_SIZE)
        #Generate a minibatch
        batch_data = train_dataset[offset:(offset+BATCH_SIZE),:]
        batch_labels = train_labels[offset:(offset+BATCH_SIZE),:]
        #Prepare a dictionary telling the session where to feed the minibatch
        #Key! the placeholder node of the graph to be fed; (must match the placeholder variable name givin in the graph)
        #value: numpy array to feed it
        feed_dict = {tf_train_dataset:batch_data,tf_train_labels:batch_labels}
        _,run_loss,prediction = session.run([optimizer,loss,train_prediction],feed_dict=feed_dict)
        if (step % 500 ==0):
            print("Minibatch loss at step %d: %f" % (step, run_loss))
            print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
            print("Validation accuracy: %.1f%%" % accuracy(valid_prediction.eval(), valid_labels))
    print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(),test_labels))
    
#Problem : turn into a 1-hidden layer neural network with rectified linear units (nn.relu()) and 1024 hidden nodes.
#This model should improve your validation / test accuracy.
    
graph =  tf.Graph()

with graph.as_default():
    
    #INPUT
    #Put training data in placeholvder with a training batch
    tf_train_dataset = tf.placeholder(tf.float32,shape=(BATCH_SIZE,IMAGE_SIZE*IMAGE_SIZE))
    tf_train_labels = tf.placeholder(tf.float32,shape=(BATCH_SIZE,NUMBER_OF_LABELS))
    TF_VALID_DATASET = tf.constant(valid_dataset)
    TF_TEST_DATASET = tf.constant(test_dataset)
    
    #HIDDEN LAYER
    HIDDEN_NODES = 1024
    
    hidden_weights = tf.Variable(tf.truncated_normal([IMAGE_SIZE*IMAGE_SIZE,HIDDEN_NODES]))
    hidden_biases = tf.Variable(tf.zeros([HIDDEN_NODES]))
    hidden_layer = tf.nn.relu(tf.matmul(tf_train_dataset,hidden_weights)+hidden_biases)
    
    #VARIABLES: Parameters used for the training
    
    #Weight matrix initialized with random values using a truncated normal distribution
    weights = tf.Variable(tf.truncated_normal([HIDDEN_NODES,NUMBER_OF_LABELS]))
    
    #biases initialize at 0
    biases = tf.Variable(tf.zeros([NUMBER_OF_LABELS]))
    
    #OPERATION
    
    # Training computation: Hidden_layer * Weight + biases
    logits = tf.matmul(hidden_layer,weights)+ biases
    
    # loss: Comput the softmax & cross entropy and take the average for all the training set
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels,logits=logits))
    
    #OPTIMIZER
    #Find the minimum of this loss using gradient descent
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
    
    #Predictions for the traning,validation & test data
    train_prediction = tf.nn.softmax(logits)
    
    valid_relu = tf.nn.relu(  tf.matmul(TF_VALID_DATASET, hidden_weights) + hidden_biases)
    valid_prediction = tf.nn.softmax(tf.matmul(valid_relu,weights)+biases)
   
    test_relu = tf.nn.relu( tf.matmul( TF_TEST_DATASET, hidden_weights) + hidden_biases)
    test_prediction = tf.nn.softmax(tf.matmul(test_relu, weights) + biases)

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print("Initialized")
    for step in range(NUM_STEPS):
        #Pick an offset within the randomized training data
        offset = (step*BATCH_SIZE) %(train_labels.shape[0] - BATCH_SIZE)
        #Generate a minibatch
        batch_data = train_dataset[offset:(offset+BATCH_SIZE),:]
        batch_labels = train_labels[offset:(offset+BATCH_SIZE),:]
        #Prepare a dictionary telling the session where to feed the minibatch
        #Key! the placeholder node of the graph to be fed;
        #value: numpy array to feed it
        feed_dict = {tf_train_dataset:batch_data,tf_train_labels:batch_labels}
        _,run_loss,prediction = session.run([optimizer,loss,train_prediction],feed_dict=feed_dict)
        if (step % 500 ==0):
            print("Minibatch loss at step %d: %f" % (step, run_loss))
            print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
            print("Validation accuracy: %.1f%%" % accuracy(valid_prediction.eval(), valid_labels))
    print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(),test_labels))
 