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


#%% Step 1-0 - Read signnames.csv
import csv
signnames = None
with open('../signnames.csv') as csvFile:
    reader = csv.reader(csvFile)
    # Skip header
    next(reader)
    signnames = [r[1] for r in reader]


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
softmax_op = tf.nn.softmax(logits=logits)
pred_count = 5
top5_op = tf.nn.top_k(softmax_op, k=pred_count)
# evaluations
correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(y_onehot, 1))
accuracy_op = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))


#%% Step 2-3 - Train, Validate and Test the Model
print('#%% Step 2-3 - Train, Validate and Test the Model ###')

import time
from sklearn.utils import shuffle

# Set hyper-parameters
EPOCHS = 15
BATCH_SIZE = 128
KEEP_PROB = 0.4

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
# saver = tf.train.Saver()
# with tf.Session() as sess:
#     saver.restore(sess, './train_data/trafficsign_train_002')
#
#     # Validate the networks against validation set
#     validation_accuracy = evaluate(X_valid_proc, y_valid)
#     print('Validation Accuracy = {:.6f}'.format(validation_accuracy))


### Step 3: Test a Model on New Images ###

#%% Read new traffic sign images and resize them
import os
import matplotlib.pyplot as plt

new_img_dir = '../image/'
image_files = sorted(os.listdir(new_img_dir))
new_img_count = len(image_files)
new_images = []
X_new, y_new = [], []
for image_name in image_files:
    # Read an image file
    img = cv2.imread(new_img_dir + image_name)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    new_images.append(img_rgb)
    # Resize the image file
    img_resized = cv2.resize(img_rgb, dsize=(32, 32), interpolation=cv2.INTER_LINEAR)
    X_new.append(img_resized)
    # Determine the traffic sign class
    img_class = int(image_name.split('.')[0])
    y_new.append(img_class)
# Preprocess images
y_new = np.array(y_new)
X_new_proc = preprocess(X_new)

# Display new images
img_row = 3
img_col = 3
plt.figure(figsize=(10, 10))
for i in range(new_img_count):
    plt.subplot(img_row, img_col, i + 1)
    img_label = y_new[i]
    plt.title('{} - {:.15}'.format(image_files[i], signnames[img_label]))
    plt.imshow(new_images[i])
plt.show()

plt.figure().suptitle('New Traffic Sign Images - Resized to 32x32')
for i in range(new_img_count):
    plt.subplot(img_row, img_col, i + 1)
    img_label = y_new[i]
    plt.title('{} - {}'.format(img_label, signnames[img_label]))
    plt.imshow(X_new[i])
plt.show()

plt.figure().suptitle('New Traffic Sign Images - Preprocessed')
for i in range(new_img_count):
    plt.subplot(img_row, img_col, i + 1)
    img_label = y_new[i]
    plt.title('{} - {}'.format(img_label, signnames[img_label]))
    plt.imshow(X_new_proc[i].squeeze(), cmap='gray')
plt.show()


#%% Evaluate new images
train_data_file = './train_data/trafficsign_train_001'
saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, train_data_file)

    # Validate the networks against validation set
    new_img_accuracy, top5_pred = evaluate(X_new_proc, y_new)

print('New Image Accuracy = {:.2%}'.format(new_img_accuracy))


#%% Display predictions for new images
plt.figure(figsize=(10, 10))

for i in range(new_img_count):
    plt.subplot(img_row, img_col, i + 1)
    img_label = y_new[i]
    label_pred = top5_pred.indices[i][0]
    correct = 'CORRECT' if label_pred == img_label else 'WRONG'
    plt.title('{} - {:.20}\nPrediction: {} ({:.2%}) {}'.format( \
            image_files[i], signnames[img_label], label_pred, top5_pred.values[i][0], correct))
    plt.imshow(X_new_proc[i].squeeze(), cmap='gray')
plt.tight_layout()
plt.show()


#%% Print prediction summary - top 5 softmax probabilities

for i in range(new_img_count):
    img_label = y_new[i]
    signname = signnames[img_label]
    predictions = list(map(lambda x: '{}-{:.20} ({:.2%})'.format(x[0], signnames[x[0]], x[1]), \
        list(zip(top5_pred.indices[i], top5_pred.values[i]))))
    # print([i for i in predictions])
    print(predictions)


#%% Display prediction summary - top 5 softmax probabilities
import pandas as pd

labels_new = np.array([[i] * pred_count for i in y_new]).flatten()
predictions_new = top5_pred.indices.flatten()
index_tuple = list(zip(
    np.array([[i] * pred_count for i in image_files]).flatten(),
    ['{:.10}'.format(signnames[i]) for i in labels_new],
    list(range(1, pred_count + 1)) * new_img_count))
pd_index = pd.MultiIndex.from_tuples(index_tuple, names=['FileName', 'Label', 'Rank'])
df = pd.DataFrame({
        'Prediction': ['{:.10} ({})'.format(signnames[i], i) for i in predictions_new],
        'Confidence': ['{:.2%}'.format(i) for i in top5_pred.values.flatten()],
        'Correct': np.equal(labels_new, predictions_new)
    },
    index=pd_index)
print(df[['Prediction', 'Confidence', 'Correct']])