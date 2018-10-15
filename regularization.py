# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 15:33:23 2018

@author: nicolas
"""

from __future__ import print_function

import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle

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

NUM_STEPS = 3001
#######################

def testAccuracy(graph):
    
    for beta in BETA_VALUES:
        with tf.Session(graph = graph) as session:
            tf.global_variables_initializer().run()
            for step in range(NUM_STEPS):
                #get the offset to generate batch from data
                offset = (step*BATCH_SIZE) % (train_labels.shape[0]-BATCH_SIZE)
                #generate batch
                batch_data = train_dataset[offset:(offset + BATCH_SIZE),:]
                batch_labels = train_labels[offset:(offset+BATCH_SIZE),:]
                #Prepare the dictionary by telling where to feed the minibatch
                #key: placeholder node of the graph to be fed (must match variable names given in graph)
                #value: numpy array to feed the node with
                feed_dict = {tf_train_dataset:batch_data,tf_train_labels:batch_labels,beta_regularization:beta}
                _,l,predictions = session.run([optimizer,loss,train_prediction],feed_dict = feed_dict)
            print("L2 regularization(beta=%.5f) Test accuracy: %.1f%%" % 
                (beta, accuracy(test_prediction.eval(), test_labels)))    
            accuracy_values.append(accuracy(test_prediction.eval(),test_labels))
    print('Best beta=%f, accuracy=%.1f%%' % (BETA_VALUES[np.argmax(accuracy_values)], max(accuracy_values)))

        
#####################  
accuracy_values = []
"""testAccuracy(graph)"""

"""
plt.semilogx(BETA_VALUES, accuracy_values)
plt.xlabel("Beta value for regularization")
plt.ylabel("Accuracy value")
plt.grid(True)
plt.title('Test accuracy by regularization (logistic)')
plt.show()            
     """    
BEST_BETA = 0.001438 #found after running accuracy_values(graph)
#neural network model
HIDDEN_SIZE = 1024
graph = tf.Graph()
with graph.as_default():
    #INPUTS
    tf_train_dataset = tf.placeholder(tf.float32,shape=(BATCH_SIZE,IMAGE_SIZE*IMAGE_SIZE))
    tf_train_labels = tf.placeholder(tf.float32,shape=(BATCH_SIZE,NUM_LABELS))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)
    beta_regularization = tf.placeholder(tf.float32)
    
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
    loss = loss+beta_regularization*(tf.nn.l2_loss(weights1)+tf.nn.l2_loss(biases1)+tf.nn.l2_loss(weights2)+tf.nn.l2_loss(biases2))
    
    #Optimizer
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
    
    #Predictions
    train_prediction = tf.nn.softmax(logits)
    layer1_valid = tf.nn.relu(tf.matmul(tf_valid_dataset,weights1)+biases1)
    valid_logits = tf.matmul(layer1_valid,weights2)+biases2
    valid_prediction = tf.nn.softmax(valid_logits)
    
    layer1_test = tf.nn.relu(tf.matmul(tf_test_dataset,weights1)+biases1)
    test_logits = tf.matmul(layer1_test,weights2)+biases2
    test_prediction = tf.nn.softmax(test_logits)


accuracy_values = []
"""testAccuracy(graph)"""


#PROBLEM 2 demonstrate overfitting by restraining  data to few batches
small_batch_size = BATCH_SIZE*5
small_train_dataset = train_dataset[:small_batch_size,:]
small_train_labels = train_labels[:small_batch_size,:]
#print('Training set',small_train_dataset,small_train_labels)

def overfitGraph(graph):
    with tf.Session(graph=graph) as session:
        tf.global_variables_initializer().run()
        for step in range(NUM_STEPS):
            offset = (step * BATCH_SIZE) % (small_train_labels.shape[0] - BATCH_SIZE)
            # Generate batch
            batch_data = small_train_dataset[offset:(offset + BATCH_SIZE), :]
            batch_labels = small_train_labels[offset:(offset + BATCH_SIZE), :]
            # Prepare a dictionary telling the session where to feed the minibatch.
            feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels, beta_regularization: BEST_BETA}
            _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
            if (step % 500 == 0):
                print("Minibatch loss at step %d: %f" % (step, l))
                print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
                print("Validation accuracy: %.1f%%" % accuracy(
                    valid_prediction.eval(), valid_labels))
        print("Overfitting with small dataset Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))
#Smaller accuracy on validation (77% but 100% on test)
"""overfitGraph(graph)"""

#Problem 3 Introduce Droupout on the hidden layer 
#dropout method: put halfs of the values at 0

graph = tf.Graph()

with graph.as_default():
    # Input 
    
    tf_train_dataset = tf.placeholder(tf.float32, shape=(BATCH_SIZE, IMAGE_SIZE * IMAGE_SIZE))
    tf_train_labels = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_LABELS))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)
    beta_regularization = tf.placeholder(tf.float32)

    # Variables.
    W1 = tf.Variable(tf.truncated_normal([IMAGE_SIZE * IMAGE_SIZE, HIDDEN_SIZE]))
    B1 = tf.Variable(tf.zeros([HIDDEN_SIZE]))

    W2 = tf.Variable(tf.truncated_normal([HIDDEN_SIZE, NUM_LABELS]))
    B2 = tf.Variable(tf.zeros([NUM_LABELS]))

    # Training computation.
    y1 = tf.nn.relu(tf.matmul(tf_train_dataset, W1) + B1)
    y1 = tf.nn.dropout(y1, 0.5)  # Dropout
    logits = tf.matmul(y1, W2) + B2

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=tf_train_labels))

    loss = loss + beta_regularization * (tf.nn.l2_loss(W1) + tf.nn.l2_loss(B1) + tf.nn.l2_loss(W2) + tf.nn.l2_loss(B2))

    #Optimizer
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
    
    #Prediction
    train_prediction = tf.nn.softmax(logits)
    y1_valid = tf.nn.relu(tf.matmul(tf_valid_dataset, W1) + B1)
    valid_logits = tf.matmul(y1_valid, W2) + B2
    valid_prediction = tf.nn.softmax(valid_logits)

    y1_test = tf.nn.relu(tf.matmul(tf_test_dataset, W1) + B1)
    test_logits = tf.matmul(y1_test, W2) + B2
    test_prediction = tf.nn.softmax(test_logits)

def trainGraph(graph):
    third = round(NUM_STEPS/2)
    with tf.Session(graph=graph) as session:
        tf.global_variables_initializer().run()
        for step in range(NUM_STEPS):
            offset = (step * BATCH_SIZE) % (train_labels.shape[0] - BATCH_SIZE)
            # Generate batch
            batch_data = train_dataset[offset:(offset + BATCH_SIZE), :]
            batch_labels = train_labels[offset:(offset + BATCH_SIZE), :]
            # Prepare a dictionary telling the session where to feed the minibatch.
            feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels, beta_regularization: BEST_BETA}
            _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
            if (step % 500 == 0):
                print("Minibatch loss at step %d: %f" % (step, l))
                print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
                print("Validation accuracy: %.1f%%" % accuracy(
                    valid_prediction.eval(), valid_labels))
            if(step % third==0 or step==1000):
                print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels)) 
        print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))

print("Train graph 1")
#trainGraph(graph)
"""
overfitGraph(graph)"""
#train & overfit accuracy is the same (around 85%)-> No overfit

#Problem 4 Train a model, use learning decay

graph = tf.Graph()

#Size max: Number of samples in training dataset(*number of classes ?)/(a*(Number of input+number of outputs))
# a between 2 and 10
BATCH_SIZE = 128
"""
HIDDEN_SIZE_1 = 1024
HIDDEN_SIZE_2 = 256
HIDDEN_SIZE_3 = 128
"""
HIDDEN_SIZE_1 = 700
HIDDEN_SIZE_2 = 300
HIDDEN_SIZE_3 = 128
print(HIDDEN_SIZE_1,HIDDEN_SIZE_2,HIDDEN_SIZE_3)
#Decays params
BASE_LEARNING_RATE = 0.4
DECAY_STEP=4000
DECAY_RATE=0.65
with graph.as_default():
     # Input data. For the training data, we use a placeholder that will be fed
  # at run time with a training minibatch.
  tf_train_dataset = tf.placeholder(tf.float32,
                                    shape=(BATCH_SIZE, IMAGE_SIZE * IMAGE_SIZE))
  tf_train_labels = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_LABELS))
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)
  beta_regularization = tf.placeholder(tf.float32)
  global_step = tf.Variable(0)

  # Variables.
  W1 = tf.Variable(tf.truncated_normal([IMAGE_SIZE * IMAGE_SIZE, HIDDEN_SIZE_1],stddev=np.sqrt(2.0 / (IMAGE_SIZE * IMAGE_SIZE))))
  B1 = tf.Variable(tf.zeros([HIDDEN_SIZE_1]))
  
  W2 = tf.Variable(tf.truncated_normal([HIDDEN_SIZE_1, HIDDEN_SIZE_2], stddev=np.sqrt(2.0 / HIDDEN_SIZE_1)))
  B2 = tf.Variable(tf.zeros([HIDDEN_SIZE_2]))
  
  W3 = tf.Variable(tf.truncated_normal([HIDDEN_SIZE_2, HIDDEN_SIZE_3], stddev=np.sqrt(2.0 / HIDDEN_SIZE_2)))
  B3 = tf.Variable(tf.zeros([HIDDEN_SIZE_3]))
  
  W_out = tf.Variable(tf.truncated_normal([HIDDEN_SIZE_3, NUM_LABELS], stddev=np.sqrt(2.0 / HIDDEN_SIZE_3)))
  B_out = tf.Variable(tf.zeros([NUM_LABELS]))
  
  # Training computation.
  layer_1 = tf.nn.relu(tf.matmul(tf_train_dataset, W1) + B1)
  #layer_1 = tf.nn.dropout(layer_1, 0.5)
  
  layer_2 = tf.nn.relu(tf.matmul(layer_1, W2) + B2)
  #layer_2 = tf.nn.dropout(layer_2, 0.5)
    
  layer_3 = tf.nn.relu(tf.matmul(layer_2, W3) + B3)
  #layer_3 = tf.nn.dropout(layer_3, 0.5)
  logits = tf.matmul(layer_3, W_out) + B_out
  
  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=tf_train_labels))
  loss = loss + beta_regularization * (tf.nn.l2_loss(W1) + tf.nn.l2_loss(B1) + tf.nn.l2_loss(W2) + tf.nn.l2_loss(B2))
  
  # Optimizer.
  learning_rate = tf.train.exponential_decay(BASE_LEARNING_RATE, global_step, DECAY_STEP, DECAY_RATE, staircase=True)
  optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
  
  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(logits)
  layer_1_valid = tf.nn.relu(tf.matmul(tf_valid_dataset, W1) + B1)
  layer_2_valid = tf.nn.relu(tf.matmul(layer_1_valid, W2) + B2)
  layer_3_valid = tf.nn.relu(tf.matmul(layer_2_valid, W3) + B3)
  valid_prediction = tf.nn.softmax(tf.matmul(layer_3_valid, W_out) + B_out)
  layer_1_test = tf.nn.relu(tf.matmul(tf_test_dataset, W1) + B1)
  layer_2_test = tf.nn.relu(tf.matmul(layer_1_test, W2) + B2)
  layer_3_test = tf.nn.relu(tf.matmul(layer_2_test, W3) + B3)
  test_prediction = tf.nn.softmax(tf.matmul(layer_3_test, W_out) + B_out)

print("Training Problem 4 graph")
#NUM_STEPS = 18001
trainGraph(graph)
#94.2% accuracy