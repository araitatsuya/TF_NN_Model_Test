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
#
#
#       

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

# Directry
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
print('*----------------*')
print(train_outcome)

# Argmax Class (_label_cls)
train_labels_cls = np.array([label.argmax() for label in train_labels])
## This is also a numpy array. Shape = (18,) ***

##############################
#
# TensorFlow Graph
#
##############################
## Saver Before TF Run
saver = tf.train.Saver()
# Dir Name
save_dir = 'AT1_Real_Test1_vol5/'
# If Dir does not exist, make it. 
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Path to the DIR (Saver file)
save_path = os.path.join(save_dir, 'best_validation')


## placeholder
# Training Data
x = tf.placeholder(tf.float32, [None, 6])

# True Outcome
y_true = tf.placeholder(tf.float32, [None, 3]) ### Added one more colmun

# True Class
y_true_cls = tf.placeholder(tf.int64, [None])

# Beta Regularization
beta_regul = tf.placeholder(tf.float32) ############### Vol 4

## Variables to be optimized
# Variables.

# # of hidden Nodes
num_hidden_nodes = 5

# Layer 1
weights1 = tf.Variable(tf.truncated_normal([6, num_hidden_nodes]))
biases1 = tf.Variable(tf.zeros([num_hidden_nodes]))

# Layer 2
weights2 = tf.Variable(tf.truncated_normal([num_hidden_nodes, 3])) ### Added one more colmun
biases2 = tf.Variable(tf.zeros([3])) ### Added one more colmun

## Model

# Training computation.
lay1_train = tf.nn.relu(tf.matmul(x, weights1) + biases1) # First Layer + ReLu filter
logits = tf.matmul(lay1_train, weights2) + biases2

# Class Prediction
y_pred = tf.nn.softmax(logits[:,0:2])
# Argmax Class
y_pred_cls = tf.argmax(y_pred[:,0:2], dimension=1)

## Cost_function to be optimized 
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits[:,0:2], labels=y_true[:,0:2]) + \
                      beta_regul * (tf.nn.l2_loss(weights1) + tf.nn.l2_loss(weights2)) + \
                      tf.reduce_sum(tf.multiply(y_pred[:,0],tf.multiply(tf.subtract(logits[:,2],y_true[:,2]),tf.subtract(logits[:,2],y_true[:,2])))) \
                      )
cost_f = tf.cast(cost, tf.float32)

## Optimization method
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(cost)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
#optimizer = tf.train.AdagradOptimizer(1.0).minimize(cost)

## Performance measures: Accuracy
#
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
#
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

############################
#
# TensorFlow Run
#
############################

## Create TensorFlow session
session = tf.Session()
## Initialize Variables
session.run(tf.global_variables_initializer())

def optimize(num_iterations):
    for i in range(num_iterations):
        feed_dict_train = {x : train_dataset, y_true: train_outcome, y_true_cls: train_labels_cls, beta_regul : 1e-3}
        opt_, accuracy_tf, cost_tf = session.run([optimizer, accuracy, cost_f], feed_dict=feed_dict_train)
        print('Steps: %d, Loss = %.5f, Acc = %.2f' %(i, cost_tf, accuracy_tf))


### accuracy
feed_dict_test = {x : train_dataset, y_true: train_outcome, y_true_cls: train_labels_cls, beta_regul : 1e-3}
def print_accuracy():
    # Use TensorFlow to compute the accuracy.
    acc = session.run(accuracy, feed_dict=feed_dict_test)
    # Print the accuracy.
    print("Accuracy on test-set: {0:.1%}".format(acc))

### With this Dataset, the initial answer should be ~ 50%
print_accuracy()

optimize(num_iterations=100)
print_accuracy()

## Weight Visualization
#w = session.run(weights)
#w = session.run(weights1)
#plt.imshow(w)
#plt.show

session.run(tf.global_variables_initializer())
optimize(num_iterations=100000)
logits_np = session.run(logits, feed_dict = feed_dict_test)
logits_np 

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
#        Tatsuya J. Arai     UTSW
#           07/18/2017
#
#
#       * L2 distance of survival days with or without class prediction. 
#           "If" overshoot -> Ignore. "If" undershoot -> Penalize
#       * Input parameters
#
#







