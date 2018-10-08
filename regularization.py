# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 15:33:23 2018

@author: nicolas
"""

from __future__ import print_function

import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
import matplotlib.pyplot as plt

pickle_file='notMNIST.pickle'

with open(pickle_file,'rb') as f:
    save = pickle.load(f)
    train_dataset = save['train_dataset']
    train_labels = save['train_labels']
    valid_dataset = save['valid_dataset']
    valid_labels = save['valid_labels']
    test_dataset = save['test_dataset']
    test_labels = save['test_labels']
    del save
    print('Training set', train_dataset.shape, train_labels.shape)
    print('Validation set', valid_dataset.shape, valid_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)

IMAGE_SIZE = 28
NUM_LABELS=10

def reformat(dataset,labels):
    dataset = dataset.reshape((-1,IMAGE_SIZE*IMAGE_SIZE)).astype(np.float32)
# Map 1 to [0.0, 1.0, 0.0 ...], 2 to [0.0, 0.0, 1.0 ...]
    labels = (np.arange(NUM_LABELS)==labels[:,None]).astype(np.float32)
    return dataset,labels

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

def accuracy (predictions,labels):
    return(100.0*np.sum(np.argmax(predictions,1)==np.argmax(labels,1))/predictions.shape[0])

#problem 1 L2 regularization for logistic & neural network

BETA_VALUES = np.logspace(-4,-2,20)
BATCH_SIZE=128
accuracy_values = []

#logistic model (see StochasticGradientNeuralNetwork)
graph = tf.Graph()
with graph.as_default():
    
    #INPUTS data
    tf_train_dataset = tf.placeholder(tf.float32,shape=(BATCH_SIZE,IMAGE_SIZE*IMAGE_SIZE))
    tf_train_labels = tf.placeholder(tf.float32,shape=(BATCH_SIZE,NUM_LABELS))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)
    beta_regularization = tf.placeholder(tf.float32)
    
    #VARIABLES
    #initlialize with random normal distribution variables 2D matrix
    weights = tf.Variable(tf.truncated_normal([IMAGE_SIZE*IMAGE_SIZE,NUM_LABELS]))
    #Initialize bias
    biases=tf.Variable(tf.zeros([NUM_LABELS]))
    
    #OPERATIONS
    logits = tf.matmul(tf_train_dataset,weights)+biases
    
    #Add the L2 regularization, compute L2 loss with nn.l2_loss(matrix)
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,labels=tf_train_labels)) + beta_regularization * tf.nn.l2_loss(weights)
 
    #OPTIMIZER
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
    
    #PREDICTIONS
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(tf.matmul(tf_valid_dataset,weights)+biases)
    test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset,weights)+biases)

NUM_STEP = 3001
#######################

def testAccuracy(graph):
    accuracy_values = []
    for beta in BETA_VALUES:
        with tf.Session(graph = graph) as session:
            tf.global_variables_initializer().run()
            for step in range(NUM_STEP):
                #get the offset to generate batch from data
                offset = (step*BATCH_SIZE) % (train_labels.shape[0]-BATCH_SIZE)
                #generate batch
                batch_data = train_dataset[offset:(offset + BATCH_SIZE),:]
                batch_labels = train_labels[offset:(offset+BATCH_SIZE),:]
                #Prepare the dictionary by telling where to feed the minibatch
                #key: placeholder node of the graph to be fed
                #value: numpy array to feed the node with
                feed_dict = {tf_train_dataset:batch_data,tf_train_labels:batch_labels,beta_regularization:beta}
                _,l,predictions = session.run([optimizer,loss,train_prediction],feed_dict = feed_dict)
            print("L2 regularization(beta=%.5f) Test accuracy: %.1f%%" % 
                (beta, accuracy(test_prediction.eval(), test_labels)))    
            accuracy_values.append(accuracy(test_prediction.eval(),test_labels))        
        
#####################   
testAccuracy(graph)

print('Best beta=%f, accuracy=%.1f%%' % (BETA_VALUES[np.argmax(accuracy_values)], max(accuracy_values)))

plt.semilogx(BETA_VALUES, accuracy_values)
plt.xlabel("Beta value for regularization")
plt.ylabel("Accuracy value")
plt.grid(True)
plt.title('Test accuracy by regularization (logistic)')
plt.show()            
         
#neural network model
HIDDEN_SIZE = 1024
graph = tf.Graph()
with graph.as_default():
    #INPUTS
    tf_train_dataset = tf.placeholder(tf.float32,shape=(BATCH_SIZE,IMAGE_SIZE*IMAGE_SIZE))
    tf_train_labels = tf.placeholder(tf.float32,shape=(BATCH_SIZE,NUM_LABELS))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)
    tf_beta = tf.placeholder(tf.float32)
    
    #VARIABLES
    #Weights & biases of the first layer (hidden)
    weights1 = tf.Variable(tf.truncated_normal([IMAGE_SIZE*IMAGE_SIZE,HIDDEN_SIZE]))
    biases1 = tf.Variable(tf.zeros([HIDDEN_SIZE]))
    #Weights& biases of the second layers 
    weights2 = tf.Variable(tf.truncated_normal([HIDDEN_SIZE,NUM_LABELS]))
    biases2 = tf.Variable(tf.zeros([NUM_LABELS]))
    
    #COMPUTATION:
    #data goes through first layer (hidden layer) then second layer for final result
    first_layer_output = tf.nn.relu(tf.matmul(tf_train_dataset,weights1)+ biases1)
    logits = tf.matmul(first_layer_output,weights2)+biases2
    
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,labels=tf_train_labels))
    loss = loss+tf_beta*(tf.nn.l2_loss(weights1)+tf.nn.l2_loss(biases1)+tf.nn.l2_loss(weights2)+tf.nn.l2_loss(biases2))
    
    #Optimizer
    tf.train.GradientDescentOptimizer(0.5).minimize(loss)
    
    #Predictions
    train_prediction = tf.nn.softmax(logits)
    layer1_valid = tf.nn.relu(tf.matmul(tf_valid_dataset,weights1)+biases1)
    valid_logits = tf.matmul(layer1_valid,weights2)+biases2
    valid_prediction = tf.nn.softmax(valid_logits)
    
    layer1_test = tf.nn.relu(tf.matmul(tf_test_dataset,weights1)+biases1)
    test_logits = tf.matmul(layer1_test,weights2)+biases2
    test_prediction = tf.nn.softmax(test_logits)


testAccuracy(graph)

print('Best beta=%f, accuracy=%.1f%%' % (BETA_VALUES[np.argmax(accuracy_values)], max(accuracy_values)))


#PROBLEM 2 demonstrate overfitting by restraining  data to few batches