# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 19:13:16 2018

@author: nicolas
"""

from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range



PICKLE_FILE = 'notMNIST.pickle'

with open(PICKLE_FILE, 'rb') as f:
  save = pickle.load(f)
  train_dataset = save['train_dataset']
  train_labels = save['train_labels']
  valid_dataset = save['valid_dataset']
  valid_labels = save['valid_labels']
  test_dataset = save['test_dataset']
  test_labels = save['test_labels']
  del save  # hint to help gc free up memory
  print('Training set', train_dataset.shape, train_labels.shape)
  print('Validation set', valid_dataset.shape, valid_labels.shape)
  print('Test set', test_dataset.shape, test_labels.shape)
  
  
IMAGE_SIZE = 28
NUM_LABELS = 10
NUM_CHANNELS = 1 # grayscale

def reformat(dataset, labels):
  dataset = dataset.reshape((-1, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)).astype(np.float32)
  labels = (np.arange(NUM_LABELS) == labels[:,None]).astype(np.float32)
  return dataset, labels
  
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)


def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

BATCH_SIZE = 16
PATCH_SIZE = 5
DEPTH = 16
NUM_HIDDEN = 64

graph = tf.Graph()

with graph.as_default():

  # Input data.
  tf_train_dataset = tf.placeholder(    tf.float32, shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
  tf_train_labels = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_LABELS))
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)
  
  # Variables.
  layer1_weights = tf.Variable(tf.truncated_normal( [PATCH_SIZE, PATCH_SIZE, NUM_CHANNELS, DEPTH], stddev=0.1))
  layer1_biases = tf.Variable(tf.zeros([DEPTH]))
  
  layer2_weights = tf.Variable(tf.truncated_normal([PATCH_SIZE, PATCH_SIZE, DEPTH, DEPTH], stddev=0.1))
  layer2_biases = tf.Variable(tf.constant(1.0, shape=[DEPTH]))
  
  # // return integer after division
  layer3_weights = tf.Variable(tf.truncated_normal([IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * DEPTH, NUM_HIDDEN], stddev=0.1))
  layer3_biases = tf.Variable(tf.constant(1.0, shape=[NUM_HIDDEN]))
  
  layer4_weights = tf.Variable(tf.truncated_normal([NUM_HIDDEN, NUM_LABELS], stddev=0.1))
  layer4_biases = tf.Variable(tf.constant(1.0, shape=[NUM_LABELS]))
  
  # Convolutional Model.
  def model(data):
    stride_value= 2
    #Problem 1: replace strides by max pooling of stride 2 & kernel size 2 
    #conv = tf.nn.conv2d(data, layer1_weights, [1, stride_value, stride_value, 1], padding='SAME')
    conv = tf.nn.conv2d(data, layer1_weights, [1, 1, 1, 1], padding='SAME')
    hidden = tf.nn.relu(conv + layer1_biases)
    pool = tf.nn.max_pool(hidden,[1,stride_value,stride_value,1],[1,stride_value,stride_value,1],padding='SAME')
    
    #Problem 1
    #conv = tf.nn.conv2d(hidden, layer2_weights, [1, stride_value, stride_value, 1], padding='SAME')
    conv = tf.nn.conv2d(pool, layer2_weights, [1, 1, 1, 1], padding='SAME')
    hidden = tf.nn.relu(conv + layer2_biases)
    pool = tf.nn.max_pool(hidden,[1,stride_value,stride_value,1],[1,stride_value,stride_value,1],padding='SAME')

    #Problem 1
    #shape = hidden.get_shape().as_list()
    shape = pool.get_shape().as_list()
     #Problem 1
    #reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
    reshape = tf.reshape(pool, [shape[0], shape[1] * shape[2] * shape[3]])

    hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
    
    return tf.matmul(hidden, layer4_weights) + layer4_biases
  
  # Training computation.
  logits = model(tf_train_dataset)
  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf_train_labels, logits=logits))
    
  # Optimizer.
  optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
  
  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(logits)
  valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
  test_prediction = tf.nn.softmax(model(tf_test_dataset))

#train the model
NUM_STEPS = 3001

def trainGraph(graph):
    with tf.Session(graph=graph) as session:
      tf.global_variables_initializer().run()
      print('Initialized')
      #Epoch
      for step in range(NUM_STEPS):
        offset = (step * BATCH_SIZE) % (train_labels.shape[0] - BATCH_SIZE)
        batch_data = train_dataset[offset:(offset + BATCH_SIZE), :, :, :]
        batch_labels = train_labels[offset:(offset + BATCH_SIZE), :]
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
        _, l, predictions = session.run(
          [optimizer, loss, train_prediction], feed_dict=feed_dict)
        if (step % 100 == 0):
          print('Minibatch loss at step %d: %f' % (step, l))
          print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
          print('Validation accuracy: %.1f%%' % accuracy(
            valid_prediction.eval(), valid_labels))
      print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))

#Problem 1: 91.3%
"""trainGraph(graph)"""

#Problem 2: add dropout, learning rate decay... use leNet5


graph = tf.Graph()

BATCH_SIZE = 32
DEPTH = 32
PATCH_SIZE = 5

train_size = train_labels.shape[0]
BASE_LEARNING_RATE = 0.01
DECAY_STEP=train_size
DECAY_RATE=0.97

NUM_HIDDEN = 512
SEED=42

with graph.as_default():

  # Input data.
  tf_train_dataset = tf.placeholder(tf.float32, shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
  tf_train_labels = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_LABELS))
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)
  global_step = tf.Variable(0)
  # Variables.
  #5x5 filter depth: 32 
  layer1_weights = tf.Variable(tf.truncated_normal([PATCH_SIZE, PATCH_SIZE, NUM_CHANNELS, DEPTH], stddev=0.1,seed=SEED))
  layer1_biases = tf.Variable(tf.zeros([DEPTH]))
  
  layer2_weights = tf.Variable(tf.truncated_normal([PATCH_SIZE, PATCH_SIZE, DEPTH, 2*DEPTH],stddev=0.1,seed=SEED))
  layer2_biases = tf.Variable(tf.constant(0.1, shape=[2*DEPTH]))
  
  layer3_weights = tf.Variable(tf.truncated_normal([IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * 2*DEPTH, NUM_HIDDEN],stddev=0.1,seed=SEED))
  layer3_biases = tf.Variable(tf.constant(0.1, shape=[NUM_HIDDEN]))
  
  layer4_weights = tf.Variable(tf.truncated_normal([NUM_HIDDEN, NUM_LABELS],stddev=0.1,seed=SEED))
  layer4_biases = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS]))

  def model(data, train=False):
    stride_size=2
    conv = tf.nn.conv2d(data,layer1_weights,strides=[1, 1, 1, 1],padding='SAME')
    # Bias and rectified linear non-linearity. bias add: add bias even if different types
    hidden = tf.nn.relu(tf.nn.bias_add(conv, layer1_biases))
    pool = tf.nn.max_pool(hidden,ksize=[1, stride_size, stride_size, 1],strides=[1, stride_size, stride_size, 1],
                          padding='SAME')
                          
    conv = tf.nn.conv2d(pool,layer2_weights,strides=[1, 1, 1, 1],
                        padding='SAME')
    hidden = tf.nn.relu(tf.nn.bias_add(conv, layer2_biases))
    pool = tf.nn.max_pool(hidden,ksize=[1, stride_size, stride_size, 1],strides=[1, stride_size, stride_size, 1],
                          padding='SAME')
    
    pool_shape = pool.get_shape().as_list()
    reshape = tf.reshape(pool,[pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
   
    hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
    if train:
      hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)
    return tf.matmul(hidden, layer4_weights) + layer4_biases

  # Training computation: logits + cross-entropy loss.
  logits = model(tf_train_dataset, True)
  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=tf_train_labels))

  # Regularization 
  regularizers = (tf.nn.l2_loss(layer3_weights) + tf.nn.l2_loss(layer3_biases) +tf.nn.l2_loss(layer4_weights) + tf.nn.l2_loss(layer4_biases))

  loss += 5e-4 * regularizers

  # Optimizer.

  learning_rate = tf.train.exponential_decay(BASE_LEARNING_RATE,global_step, DECAY_STEP, DECAY_RATE,staircase=True)
      
  optimizer = tf.train.MomentumOptimizer(learning_rate,0.9).minimize(loss,global_step=global_step) 

 # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(logits)
  valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
  test_prediction = tf.nn.softmax(model(tf_test_dataset))

#Problem 2:
trainGraph(graph)
#94.4%