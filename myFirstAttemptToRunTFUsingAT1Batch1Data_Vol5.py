#
#   AT1 Batch 1 Data
#   O:\Research\Pre-Clinical\Current team\TatsuyaArai\MATLAB\AT1_ST
#   MAT_Py_TF_Data_test2.mat Treated on July
#       Train_Paras (18 Tumors x 6 independent variables)
#       Label (18 Tumor x 2 outcome classes)
#       Label_1D (18 Tumors x 1D outcome class (0 or 1)) (*Probably You don't need it*)
#

#
#   Vol. 1 Linear Eq
#   Vol. 2
#       Adding Saver
#           TF Tutorial #4: Save and Restore
#   Vol. 3
#       Fully Connected Layer
#           Udacity #2
#   Vol. 4
#       Smat Loss(Cost) Calculation
#           Udacity #3 [7]~ Regularization
#       beta_regul = tf.placeholder(tf.float32)
#       loss = tf.reduce_mean(
#    tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = tf_train_labels)) + beta_regul * tf.nn.l2_loss(weights)
#       feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels, beta_regul : 1e-3}
#
#       # Training computation. [13]~
#       lay1_train = tf.nn.relu(tf.matmul(tf_train_dataset, weights1) + biases1)
#       logits = tf.matmul(lay1_train, weights2) + biases2
#       loss = tf.reduce_mean( ###***
#       tf.nn.softmax_cross_entropy_with_logits(logits = logits,labels = tf_train_labels)) + \  ###****
#                               beta_regul * (tf.nn.l2_loss(weights1) + tf.nn.l2_loss(weights2)) ###****
#
#       *** Udacity #3 [22]~ Dropout (Fully Connected)
#           Dropout
#
#           # Training computation.
#           lay1_train = tf.nn.relu(tf.matmul(tf_train_dataset, weights1) + biases1)
#           drop1 = tf.nn.dropout(lay1_train, 0.5)
#           logits = tf.matmul(drop1, weights2) + biases2
#           loss = tf.reduce_mean(
#           tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = tf_train_labels))
#
#           # Predictions for the training, validation, and test data.
#           (You don't need to use dropout for test and validation data)
#           train_prediction = tf.nn.softmax(logits)
#           lay1_valid = tf.nn.relu(tf.matmul(tf_valid_dataset, weights1) + biases1)
#           valid_prediction = tf.nn.softmax(tf.matmul(lay1_valid, weights2) + biases2)
#           lay1_test = tf.nn.relu(tf.matmul(tf_test_dataset, weights1) + biases1)
#           test_prediction = tf.nn.softmax(tf.matmul(lay1_test, weights2) + biases2)
#
#       *** Udacity #3 [24]~ Learning Rate Decay
#           global_step = tf.Variable(0)  # count the number of steps taken.
#           learning_rate = tf.train.exponential_decay(0.5, step, ...)
#           optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
#
#          # Optimizer.
#           learning_rate = tf.train.exponential_decay(0.5, global_step, 1000, 0.65, staircase=True)
#           optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
#
#   Vol. 5 (A bit messy)
#       Smarter Cost Calculation
#           * Soft Max Cross Entropy based on class prediction.
#           * If the tumor is categorized as class 0 L2 distance of predicted survival duration matters. 
#               
#       cost = tf.reduce_mean( \
#                      tf.nn.softmax_cross_entropy_with_logits(logits=logits[:,0:2], labels=y_true[:,0:2]) + \ ### Soft Max Cros Entropy: Class Prediction
#                      beta_regul * (tf.nn.l2_loss(weights1) + tf.nn.l2_loss(weights2)) + \ ### Regularization Loss
#                      tf.reduce_sum(tf.multiply(y_pred[:,0],tf.multiply(tf.subtract(logits[:,2],y_true[:,2]),tf.subtract(logits[:,2],y_true[:,2])))) \ ### Survival Days Prediction
#                      )


#
#   Some Tricks
#       Trainable_variables()
#           [var.name for var in tf.trainable_variables()]
#
#


# MAT -> (Num)py
import numpy as np
import scipy
from scipy import io

import os

# Imports (TensorFlow Tutorial)
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix

# Data Loading From .mat
MAT = scipy.io.loadmat("MAT_Py_TF_Data_test2")
train_dataset = MAT['Train_Paras'] # This is a numpy array. Shape = (18,6)
train_labels = MAT['Label'] # This is a numpy arrray. Shape = (18,2)
train_labels_1D = MAT['Label_1D'] # This is a numpy array. Shape = (18,1) ***
train_fate2 = MAT['Class_Fate'] # This is a numpy array. Shape = (18,2) ***

train_fate = train_fate2[:,1] # This is a numpy array. Shape = (18,)
train_fate.astype(np.float32)

###
train_outcome = np.zeros((18,3)) # Whatever the reason, double (())
train_outcome[:,0:2] = train_labels
train_outcome[:,2] = train_fate
train_outcome.astype(np.float32)

# Udacity...
train_dataset.astype(np.float32)
train_labels.astype(np.float32)

print(train_dataset)
print('*----------------*')
print(train_labels)
print('*----------------*')
print(train_labels_1D)

# From here, I am testing out the TensorFlow tutorial code with my data.tra

# [5] and [6]
# data.test.labels[0:5, :]
# data.test.cls = np.array([label.argmax() for label in data.test.labels])
# data.test.cls[0:5]
train_labels_1D_argmax = np.array([label.argmax() for label in train_labels])
train_labels_cls = np.array([label.argmax() for label in train_labels])
## This is also a numpy array. Shape = (18,) ***
train_labels_1D_argmax[0:18]

# [11]
## We know that MNIST images are 28 pixels in each dimension.
#img_size = 28
## Images are stored in one-dimensional arrays of this length.
#img_size_flat = img_size * img_size (***img_size_flat is equivalent of # of independent variables***)
## Tuple with height and width of images used to reshape arrays.
#img_shape = (img_size, img_size)
## Number of classes, one class for each of 10 digits.
#num_classes = 10 (***# of outcome classes***)

# TensorFlow Graph
# placeholder
# [11]
# x = tf.placeholder(tf.float32, [None, img_size_flat])
x = tf.placeholder(tf.float32, [None, 6])

# [12]
# y_true = tf.placeholder(tf.float32, [None, num_classes])
# y_true = tf.placeholder(tf.float32, [None, 2])
y_true = tf.placeholder(tf.float32, [None, 3]) ### Added one more colmun

# [13]
# y_true_cls = tf.placeholder(tf.int64, [None])
y_true_cls = tf.placeholder(tf.int64, [None])
beta_regul = tf.placeholder(tf.float32) ############### Vol 4

# Variables to be optimized
## [14]
## weights = tf.Variable(tf.zeros([img_size_flat, num_classes]))
#weights = tf.Variable(tf.zeros([6, 2]))
## [15]
## biases = tf.Variable(tf.zeros([num_classes]))
#biases = tf.Variable(tf.zeros([2]))

# Variables.
num_hidden_nodes = 5
weights1 = tf.Variable(tf.truncated_normal([6, num_hidden_nodes]))
biases1 = tf.Variable(tf.zeros([num_hidden_nodes]))
# weights2 = tf.Variable(tf.truncated_normal([num_hidden_nodes, 2]))
# biases2 = tf.Variable(tf.zeros([2]))
weights2 = tf.Variable(tf.truncated_normal([num_hidden_nodes, 3])) ### Added one more colmun
biases2 = tf.Variable(tf.zeros([3])) ### Added one more colmun

## Model
## [16]
## logits = tf.matmul(x, weights) + biases
#logits = tf.matmul(x, weights) + biases

# Training computation.
#lay1_train = tf.nn.relu(tf.matmul(tf_train_dataset, weights1) + biases1) # First Layer + ReLu filter
lay1_train = tf.nn.relu(tf.matmul(x, weights1) + biases1) # First Layer + ReLu filter
logits = tf.matmul(lay1_train, weights2) + biases2
#loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_train_labels))

# [17]
# y_pred = tf.nn.softmax(logits)
#y_pred = tf.nn.softmax(logits)
y_pred = tf.nn.softmax(logits[:,0:2]) #### This worked !!!
# [18]
# y_pred_cls = tf.argmax(y_pred, dimension=1)
y_pred_cls = tf.argmax(y_pred[:,0:2], dimension=1)

## Cost_function to be optimized
## [19]
#cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_true)
## [20]
#cost = tf.reduce_mean(cross_entropy)
#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_true))
#       loss = tf.reduce_mean( ###***
#       tf.nn.softmax_cross_entropy_with_logits(logits = logits,labels = tf_train_labels)) + \  ###****
#                               beta_regul * (tf.nn.l2_loss(weights1) + tf.nn.l2_loss(weights2)) ###****

#### As I expected, this part is hard. 
#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits[:,0:2], labels=y_true[:,0:2]) + \
#                      beta_regul * (tf.nn.l2_loss(weights1) + tf.nn.l2_loss(weights2)) +\
#                      tf.multiply(tf.subtract(logits[:,2],y_true[:,2]),tf.transpose(tf.subtract(logits[:,2],y_true[:,2]))))
#                      #tf.matmul(np.clip(logits[:,2], 0.0, np.max(train_fate)) - y_true[:,2],tf.transpose(np.clip(logits[:,2], 0.0, np.max(train_fate)) - y_true[:,2])))
#                      #tf.nn.l2_loss(np.clip(logits[:,2], 0.0, np.max(train_fate)) - y_true[:,2]))

#### As I expected, this part is hard. 
#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits[:,0:2], labels=y_true[:,0:2]) + \
#                      beta_regul * (tf.nn.l2_loss(weights1) + tf.nn.l2_loss(weights2)))
##                      tf.reduce_sum(tf.multiply(y_pred[:,0],tf.multiply(tf.subtract(logits[:,2],y_true[:,2]),tf.subtract(logits[:,2],y_true[:,2])))) \
##                      )
##                      #tf.matmul(np.clip(logits[:,2], 0.0, np.max(train_fate)) - y_true[:,2],tf.transpose(np.clip(logits[:,2], 0.0, np.max(train_fate)) - y_true[:,2])))
##                      #tf.nn.l2_loss(np.clip(logits[:,2], 0.0, np.max(train_fate)) - y_true[:,2]))

#### As I expected, this part is hard. 
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits[:,0:2], labels=y_true[:,0:2]) + \
                      beta_regul * (tf.nn.l2_loss(weights1) + tf.nn.l2_loss(weights2)) + \
                      tf.reduce_sum(tf.multiply(y_pred[:,0],tf.multiply(tf.subtract(logits[:,2],y_true[:,2]),tf.subtract(logits[:,2],y_true[:,2])))) \
                      )
##                      #tf.matmul(np.clip(logits[:,2], 0.0, np.max(train_fate)) - y_true[:,2],tf.transpose(np.clip(logits[:,2], 0.0, np.max(train_fate)) - y_true[:,2])))
##                      #tf.nn.l2_loss(np.clip(logits[:,2], 0.0, np.max(train_fate)) - y_true[:,2]))


cost_f = tf.cast(cost, tf.float32)

# Optimization method
# [21]
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(cost)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
#optimizer = tf.train.AdagradOptimizer(1.0).minimize(cost)

# Performance measures
# [22]
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
# [23]
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Saver Before TF Run
# [4 - 23]
saver = tf.train.Saver()
# [4 - 24]
save_dir = 'AT1_Real_Test1/'
# [4 - 25]
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


# [4 - 26]
save_path = os.path.join(save_dir, 'best_validation')

#saver.save(sess=session, save_path=save_path)
#saver.restore(sess=session, save_path=save_path)


# TensorFlow Run
# Create TensorFlow session
# [24]
session = tf.Session()
# Initialize Variables
# [25]
session.run(tf.global_variables_initializer())

# [27 ?] Inside [...Changed to 30]
# I don't need to use bactch now...
# This is the most important part. 
#feed_dict_train = {x: x, y_true: y_true_batch}
#session.run(optimizer, feed_dict=feed_dict_train)

def optimize(num_iterations):
    for i in range(num_iterations):
        #feed_dict_train = {x: train_dataset, y_true: train_labels}
        #feed_dict_train = {x : train_dataset, y_true: train_labels, beta_regul : 1e-3}
        #feed_dict_train = {x : train_dataset, y_true: train_outcome, y_true_cls: train_labels_cls, beta_regul : 1e-3}
        feed_dict_train = {x : train_dataset, y_true: train_outcome, y_true_cls: train_labels_cls, beta_regul : 1e-3}
        opt_, accuracy_tf, cost_tf = session.run([optimizer, accuracy, cost_f], feed_dict=feed_dict_train)
        #_ = session.run([optimizer, cost], feed_dict=feed_dict_train)
        #_, l = session.run([optimizer, cost], feed_dict=feed_dict_train)
        # Use TensorFlow to compute the Cost.
        #cost_p = session.run(cost, feed_dict=feed_dict_train)
        #print('Steps: {} {} {}'.format(i, accuracy_tf, cost_tf))
        print('Loss = %.2s' % cost_tf)
        # Print the Cost.
        #print("Cost on test-set: {0:.1}".format(l))

### accuracy
#feed_dict_test = {x: train_dataset, y_true: train_labels, y_true_cls: train_labels_cls} # This should be done with test data
#feed_dict_test = {x : train_dataset, y_true: train_labels, y_true_cls: train_labels_cls}
feed_dict_test = {x : train_dataset, y_true: train_outcome, y_true_cls: train_labels_cls, beta_regul : 1e-3}
def print_accuracy():
    # Use TensorFlow to compute the accuracy.
    acc = session.run(accuracy, feed_dict=feed_dict_test)
    
    # Print the accuracy.
    print("Accuracy on test-set: {0:.1%}".format(acc))

def print_cost():
    # Use TensorFlow to compute the Cost.
    cost_p = session.run(cost, feed_dict=feed_dict_test)
    
    # Print the Cost.
    print("Cost on test-set: {0:.1}".format(cost_p))


### With This Dataset, the answer should be ~ 50%
print_accuracy()

##### This is basically the work flow
session.run(tf.global_variables_initializer())
print_accuracy()
#w = session.run(weights)
#w = session.run(weights1)
#plt.imshow(w)
#plt.show

optimize(num_iterations=100)
print_accuracy()
#w = session.run(weights)
#w = session.run(weights1)
#plt.imshow(w)
#plt.show

optimize(num_iterations=100000)
logits_np = session.run(logits, feed_dict = feed_dict_test)

#### Save Parameters ####
saver.save(sess=session, save_path=save_path)

#### Initialize Parameters ####
session.run(tf.global_variables_initializer())
print_accuracy()
#### Restore Parameters ####
saver.restore(sess=session, save_path=save_path)
print_accuracy()

#############
#
# * More complicated patterns of loss function
#   * Regularization loss
#   * 
# * MNIST data are already flattened.
#










