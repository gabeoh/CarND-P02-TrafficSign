#%% Step 0 - Load The Data ###
print('#%% Step 0 - Load The Data ###')

# Load pickled data
import pickle

# Specify the data files for training, validation, and testing
training_file = '../data/train.p'
validation_file = '../data/valid.p'
testing_file = '../data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

assert(len(X_train) == len(y_train))
assert(len(X_valid) == len(y_valid))
assert(len(X_test) == len(y_test))


### Step 2 - Design and Test a Model Architecture

#%% Step 2-1 - Preprocess
print('#%% Step 2-1 - Preprocess ###')

import numpy as np
import cv2

### Preprocess the data here. It is required to normalize the data. Other preprocessing steps could include
### converting to grayscale, etc.
def preprocess(data):
    data_proc = data

    # Gray Scale
    data_proc = np.array(list(map(lambda x: cv2.cvtColor(x, cv2.COLOR_RGB2GRAY), data)))
    data_proc = np.expand_dims(data_proc, axis=3)

    # Normalization ([0, 255] => [-1, 1))
    data_proc = (data_proc - 128.0) / 128.0
    return data_proc

X_valid_proc = preprocess(X_valid)
X_test_proc = preprocess(X_test)


#%% Step 2-2 - Model Architecture
print('#%% Step 2-2 - Model Architecture ###')

import tensorflow as tf

mu = 0
sigma = 0.1
weights = {
    'w1': tf.Variable(tf.zeros([5, 5, 1, 12])),
    'w2': tf.Variable(tf.zeros([5, 5, 12, 32])),
    'w3': tf.Variable(tf.zeros([800, 240])),
    'w4': tf.Variable(tf.zeros([240, 120])),
    'w5': tf.Variable(tf.zeros([120, 43]))
}
biases = {
    'b1': tf.Variable(tf.zeros([12])),
    'b2': tf.Variable(tf.zeros([32])),
    'b3': tf.Variable(tf.zeros([240])),
    'b4': tf.Variable(tf.zeros([120])),
    'b5': tf.Variable(tf.zeros([43]))
}

# Build neural networks
def build_nn(x, keep_prob):
    conv_stride = [1, 1, 1, 1]
    padding = 'VALID'
    pool_kernel = [1, 2, 2, 1]
    pool_stride = [1, 2, 2, 1]

    # Layer 1 - Conv:(32, 32, 1)=>(28, 28, 6) -- ReLU -- MaxPool:=>(14, 14, 6)
    l1 = tf.nn.conv2d(x, weights['w1'], conv_stride, padding) + biases['b1']
    l1 = tf.nn.relu(l1)
    l1 = tf.nn.max_pool(l1, pool_kernel, pool_stride, padding)

    # Layer 2 - Conv:(14, 14, 6)=>(10, 10, 16) -- ReLU -- MaxPool:=>(5, 5, 16) -- Flat:=>(400)
    l2 = tf.nn.conv2d(l1, weights['w2'], conv_stride, padding) + biases['b2']
    l2 = tf.nn.relu(l2)
    l2 = tf.nn.max_pool(l2, pool_kernel, pool_stride, padding)
    l2 = tf.contrib.layers.flatten(l2)

    # Layer 3 - Full:(400)=>(120) -- ReLU
    l3 = tf.add(tf.matmul(l2, weights['w3']), biases['b3'])
    l3 = tf.nn.relu(l3)
    l3 = tf.nn.dropout(l3, keep_prob=keep_prob)

    # Layer 4 - Full:(120)=>(84) -- ReLU
    l4 = tf.add(tf.matmul(l3, weights['w4']), biases['b4'])
    l4 = tf.nn.relu(l4)
    l4 = tf.nn.dropout(l4, keep_prob=keep_prob)

    # Layer 5 - Full:(84)=>(43)
    l5 = tf.add(tf.matmul(l4, weights['w5']), biases['b5'])

    return l5

# Setup placeholders for features and labels
x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
y_onehot = tf.one_hot(y, 43)
keep_prob = tf.placeholder(tf.float32)

# Build neural networks for:
# modeling
logits = build_nn(x, keep_prob)
# evaluations
correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(y_onehot, 1))
accuracy_op = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))


#%% Step 2-3 - Train, Validate and Test the Model
print('#%% Step 2-3 - Train, Validate and Test the Model ###')

import time
from sklearn.utils import shuffle

# Set hyper-parameters
EPOCHS = 30
BATCH_SIZE = 128
KEEP_PROB = 0.4

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0.0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        index_end = offset + BATCH_SIZE
        batch_x, batch_y = X_data[offset:index_end], y_data[offset:index_end]
        accuracy = sess.run(accuracy_op, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

# Train the neural networks
saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, './train_data/trafficsign_train_002')

    # Validate the networks against validation set
    validation_accuracy = evaluate(X_valid_proc, y_valid)
    print('Validation Accuracy = {:.6f}'.format(validation_accuracy))





