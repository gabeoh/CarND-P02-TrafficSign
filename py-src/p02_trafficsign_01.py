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


#%% Step 1 - Dataset Summary & Exploration ###
print('#%% Step 1 - Dataset Summary & Exploration ###')

### Replace each question mark with the appropriate value.
### Use python, pandas or numpy methods rather than hard coding the results

import numpy as np

# Number of training examples
n_train = X_train.shape[0]

# Number of validation examples
n_valid = X_valid.shape[0]

# Number of testing examples.
n_test = X_test.shape[0]

# What's the shape of an traffic sign image?
image_shape = X_train.shape[1:3]

# How many unique classes/labels there are in the dataset.
n_classes = np.unique(np.append(np.append(y_train, y_valid), y_test)).shape[0]

print("Number of training examples =", n_train)
print("Number of validation examples =", n_valid)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

#%% Step 1-0a - Visualize dataset summary
import pandas as pd
import matplotlib.pyplot as plt

df_data_count = pd.DataFrame({
        'ImageCount': [n_train, n_valid, n_test]
    }, index=['Train', 'Validation', 'Test'])
print(df_data_count)

plt.figure()
df_data_count.plot.bar()
plt.title('Image Count for Each Dataset Type')
plt.xticks(rotation='horizontal')
#### plt.savefig('../results/count_by_dataset.png')
plt.show()

bc_train = np.bincount(y_train)
bc_valid = np.bincount(y_valid)
bc_test = np.bincount(y_test)
assert(len(bc_train) == len(bc_valid))
assert(len(bc_valid) == len(bc_test))
y_bins = range(len(bc_train))
df_data_labels = pd.DataFrame({
        'Train': bc_train,
        'Validation': bc_valid,
        'Test': bc_test
    }, index=y_bins, columns=['Train', 'Validation', 'Test'])
plt.figure()
df_data_labels.plot.bar(stacked=True)
plt.title('Occurence of Each Traffic Sign Class')
plt.xticks(rotation='vertical')
#### plt.savefig('../results/count_by_traffic_sign.png')
plt.show()

#%% Step 1-0b - Read signnames.csv
import csv
signnames = None
with open('../signnames.csv') as csvFile:
    reader = csv.reader(csvFile)
    # Skip header
    next(reader)
    signnames = [r[1] for r in reader]

#%% Step 1-1 - Visualize the Data
print('#%% Step 1-1 - Visualize the Data ###')

### Data exploration visualization code goes here.
import matplotlib.pyplot as plt
import random

def get_img_indexes_for(count_train, display_count, label=None):
    if label == None:
        return np.random.randint(0, count_train, display_count)
    else:
        indexes = [i for i, j in enumerate(y_train) if j == label]
        random.shuffle(indexes)
        return indexes

# Show a random image from training image set
img_row = 3
img_col = 3
display_count = img_row * img_col
count_train = X_train.shape[0]
img_indexes = get_img_indexes_for(count_train, display_count)
plt.figure(figsize=(10, 10))
for i in range(display_count):
    plt.subplot(img_row, img_col, i + 1)
    img_index = img_indexes[i]
    img_label = y_train[img_index]
    plt.title('#{} - {}:{:.15}'.format(img_index, img_label, signnames[img_label]))
    plt.imshow(X_train[img_index])
plt.show()


#%% Demonstrate preprocessing steps
i = 12315
img_label = y_train[i]
plt.subplot(2, 2, 1)
plt.title('#{} - {}:{:.15} (Original)'.format(i, img_label, signnames[img_label]))
plt.imshow(X_train[i])

import cv2
i_gray = cv2.cvtColor(X_train[i], cv2.COLOR_RGB2GRAY)
plt.subplot(2, 2, 2)
plt.title('#{} - {}:{:.15} (Grayscale)'.format(i, img_label, signnames[img_label]))
plt.imshow(i_gray, cmap='gray')

i_norm = (i_gray - 128.0) / 128.0
plt.subplot(2, 2, 3)
plt.title('#{} - {}:{:.15} (Normalized)'.format(i, img_label, signnames[img_label]))
plt.imshow(i_norm, cmap='gray')
plt.tight_layout()
plt.savefig('../results/preprocessing.png')
plt.show()


### Step 2 - Design and Test a Model Architecture

#%% Step 2-1 - Preprocess
print('#%% Step 2-1 - Preprocess ###')

### Preprocess the data here. It is required to normalize the data. Other preprocessing steps could include
### converting to grayscale, etc.
import cv2

def preprocess(data):
    data_proc = data

    # Gray Scale
    data_proc = np.array(list(map(lambda x: cv2.cvtColor(x, cv2.COLOR_RGB2GRAY), data)))
    data_proc = np.expand_dims(data_proc, axis=3)

    # Normalization ([0, 255] => [-1, 1))
    data_proc = (data_proc - 128.0) / 128.0
    return data_proc

X_train_proc = preprocess(X_train)
X_valid_proc = preprocess(X_valid)
X_test_proc = preprocess(X_test)

# Display processed image
plt.figure(figsize=(10, 10))
for i in range(display_count):
    plt.subplot(img_row, img_col, i + 1)
    img_index = img_indexes[i]
    img_label = y_train[img_index]
    plt.title('#{} - {}:{:.15}'.format(img_index, img_label, signnames[img_label]))
    plt.imshow(X_train_proc[img_index].squeeze(), cmap='gray')
plt.show()


#%% Step 2-2 - Model Architecture
print('#%% Step 2-2 - Model Architecture ###')

import tensorflow as tf

mu = 0
sigma = 0.1
weights = {
    'w1': tf.Variable(tf.truncated_normal([5, 5, 1, 12], mean=mu, stddev=sigma)),
    'w2': tf.Variable(tf.truncated_normal([5, 5, 12, 32], mean=mu, stddev=sigma)),
    'w3': tf.Variable(tf.truncated_normal([800, 240], mean=mu, stddev=sigma)),
    'w4': tf.Variable(tf.truncated_normal([240, 120], mean=mu, stddev=sigma)),
    'w5': tf.Variable(tf.truncated_normal([120, 43], mean=mu, stddev=sigma))
}
biases = {
    'b1': tf.Variable(tf.truncated_normal([12], mean=mu, stddev=sigma)),
    'b2': tf.Variable(tf.truncated_normal([32], mean=mu, stddev=sigma)),
    'b3': tf.Variable(tf.truncated_normal([240], mean=mu, stddev=sigma)),
    'b4': tf.Variable(tf.truncated_normal([120], mean=mu, stddev=sigma)),
    'b5': tf.Variable(tf.truncated_normal([43], mean=mu, stddev=sigma))
}

# Build neural networks
def build_nn(x, keep_prob):
    conv_stride = [1, 1, 1, 1]
    padding = 'VALID'
    pool_kernel = [1, 2, 2, 1]
    pool_stride = [1, 2, 2, 1]

    # Layer 1 - Conv:(32, 32, 1)=>(28, 28, 12) -- ReLU -- MaxPool:=>(14, 14, 12)
    l1 = tf.nn.conv2d(x, weights['w1'], conv_stride, padding) + biases['b1']
    l1 = tf.nn.relu(l1)
    l1 = tf.nn.max_pool(l1, pool_kernel, pool_stride, padding)

    # Layer 2 - Conv:(14, 14, 12)=>(10, 10, 32) -- ReLU -- MaxPool:=>(5, 5, 32) -- Flat:=>(800)
    l2 = tf.nn.conv2d(l1, weights['w2'], conv_stride, padding) + biases['b2']
    l2 = tf.nn.relu(l2)
    l2 = tf.nn.max_pool(l2, pool_kernel, pool_stride, padding)
    l2 = tf.contrib.layers.flatten(l2)

    # Layer 3 - Full:(800)=>(240) -- ReLU
    l3 = tf.add(tf.matmul(l2, weights['w3']), biases['b3'])
    l3 = tf.nn.relu(l3)
    l3 = tf.nn.dropout(l3, keep_prob=keep_prob)

    # Layer 4 - Full:(240)=>(120) -- ReLU
    l4 = tf.add(tf.matmul(l3, weights['w4']), biases['b4'])
    l4 = tf.nn.relu(l4)
    l4 = tf.nn.dropout(l4, keep_prob=keep_prob)

    # Layer 5 - Full:(120)=>(43)
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
softmax_op = tf.nn.softmax(logits=logits)
pred_count = 5
top5_op = tf.nn.top_k(softmax_op, k=pred_count)
# training
learn_rate = 0.001
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_onehot, logits=logits)
loss_op = tf.reduce_mean(cross_entropy)

# Apply L2 regularization
# beta = 0.01
# regularizer = tf.nn.l2_loss(weights['w1']) + tf.nn.l2_loss(weights['w2']) + \
#     tf.nn.l2_loss(weights['w3']) + tf.nn.l2_loss(weights['w4']) + tf.nn.l2_loss(weights['w5'])
# loss_op = tf.reduce_mean(loss_op + beta * regularizer)

optimizer = tf.train.AdamOptimizer(learning_rate=learn_rate)
training_op = optimizer.minimize(loss_op)
# evaluations
correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(y_onehot, 1))
accuracy_op = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))


#%% Step 2-3 - Train, Validate and Test the Model
print('#%% Step 2-3 - Train, Validate and Test the Model ###')

### Train your model here.
### Calculate and report the accuracy on the training and validation set.
import time
from sklearn.utils import shuffle

# Set hyper-parameters
EPOCHS = 2
BATCH_SIZE = 128
KEEP_PROB = 0.4

def train(X_data, y_data):
    num_examples = len(X_data)
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        index_end = offset + BATCH_SIZE
        batch_x, batch_y = X_data[offset:index_end], y_data[offset:index_end]
        sess.run(training_op, feed_dict={x: batch_x, y: batch_y, keep_prob: KEEP_PROB})

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0.0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        index_end = offset + BATCH_SIZE
        batch_x, batch_y = X_data[offset:index_end], y_data[offset:index_end]
        accuracy, top5_pred = sess.run([accuracy_op, top5_op], \
                feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
        total_accuracy += (accuracy * len(batch_x))
    return (total_accuracy / num_examples, top5_pred)

# Train the neural networks
train_data_file = './train_data/trafficsign_train_003'
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    start_time = time.perf_counter()
    print('Start the neural network training (at {})'.format(start_time))
    for i in range(EPOCHS):
        # Train the neural networks
        X_train_proc, y_train= shuffle(X_train_proc, y_train)
        train(X_train_proc, y_train)

        # Train the neural networks
        train_accuracy, _ = evaluate(X_train_proc, y_train)

        # Validate the networks against validation set
        validation_accuracy, _ = evaluate(X_valid_proc, y_valid)

        print('EPOCH {:<2} - Train Accuracy = {:.3f}, Validation Accuracy = {:.3f}'.format(
            i + 1, train_accuracy, validation_accuracy))
        # print('{}\t{}\t{}'.format(i+1, train_accuracy, validation_accuracy))

    end_time = time.perf_counter()
    print('Completed the neural network training in {:.3f}s (at {})'.format(end_time - start_time, end_time))
    saver.save(sess, train_data_file)


### Calculate and report the accuracy on the test set
with tf.Session() as sess:
    saver.restore(sess, train_data_file)

    # Validate the networks against test set
    test_accuracy, _ = evaluate(X_test_proc, y_test)
    print('Test Accuracy = {:.3f}'.format(test_accuracy))




