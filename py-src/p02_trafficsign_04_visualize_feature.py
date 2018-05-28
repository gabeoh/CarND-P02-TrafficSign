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

    # Layer 1 - Conv:(32, 32, 1)=>(28, 28, 12) -- ReLU -- MaxPool:=>(14, 14, 12)
    l1 = tf.nn.conv2d(x, weights['w1'], conv_stride, padding) + biases['b1']
    l1 = tf.nn.relu(l1)
    conv1 = l1
    l1 = tf.nn.max_pool(l1, pool_kernel, pool_stride, padding)

    # Layer 2 - Conv:(14, 14, 12)=>(10, 10, 32) -- ReLU -- MaxPool:=>(5, 5, 32) -- Flat:=>(800)
    l2 = tf.nn.conv2d(l1, weights['w2'], conv_stride, padding) + biases['b2']
    l2 = tf.nn.relu(l2)
    conv2 = l2
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
    logits = l5

    return logits, conv1, conv2


# Setup placeholders for features and labels
x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
y_onehot = tf.one_hot(y, 43)
keep_prob = tf.placeholder(tf.float32)

# Build neural networks for:
# modeling
logits, conv1, conv2 = build_nn(x, keep_prob)
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



#%% Step 4 - Visualize the Neural Network's State with Test Images

### Visualize your network's feature maps here.
### Feel free to use as many code cells as needed.

# image_input: the test image being fed into the network to produce the feature maps
# tf_activation: should be a tf variable name used during your training procedure that represents the calculated state of a specific weight layer
# sess: tensor flow session
# activation_min/max: can be used to view the activation contrast in more detail, by default matplot sets min and max to the actual min and max values of the output
# plt_num: used to plot out multiple different weight feature map sets on the same block, just extend the plt number for each new feature map entry

def outputFeatureMap(image_input, tf_activation, sess, file=None, activation_min=-1, activation_max=-1 ,plt_num=1):
    # Here make sure to preprocess your image_input in a way your network expects
    # with size, normalization, ect if needed
    # image_input =
    # Note: x should be the same name as your network's tensorflow data placeholder variable
    # If you get an error tf_activation is not defined it may be having trouble accessing the variable from inside a function
    activation = tf_activation.eval(session=sess,feed_dict={x : image_input})
    featuremaps = activation.shape[3]
    plt.figure(plt_num, figsize=(15,10))
    for featuremap in range(featuremaps):
        plt.subplot(4,8, featuremap+1) # sets the number of feature maps to show on each row and column
        plt.title('FeatureMap ' + str(featuremap)) # displays the feature map number
        if activation_min != -1 & activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin =activation_min, vmax=activation_max, cmap="gray")
        elif activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmax=activation_max, cmap="gray")
        elif activation_min !=-1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin=activation_min, cmap="gray")
        else:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", cmap="gray")

    plt.tight_layout()
    if (file):
        plt.savefig(file)
    plt.show()




#%% Visualize Convolution Layer Feature Maps
train_data_file = '../train_data/trafficsign_train_001'
saver = tf.train.Saver()

# Visualize Convolution Layer Feature Maps - Right-of-way at the next intersection
ii = image_files.index('11.1.jpg')
plt.imshow(X_new_proc[ii].squeeze(), cmap="gray")
plt.savefig('../results/visualize_11.png')
plt.show()
with tf.Session() as sess:
    saver.restore(sess, train_data_file)
    outputFeatureMap(X_new_proc[ii : ii + 1], conv1, sess, file='../results/visualize_11_conv1.png')
    outputFeatureMap(X_new_proc[ii : ii + 1], conv2, sess, file='../results/visualize_11_conv2.png')

# Visualize Convolution Layer Feature Maps - Roundabout mandatory
ii = image_files.index('40.1.jpg')
plt.imshow(X_new_proc[ii].squeeze(), cmap="gray")
plt.savefig('../results/visualize_40.png')
plt.show()
with tf.Session() as sess:
    saver.restore(sess, train_data_file)
    outputFeatureMap(X_new_proc[ii : ii + 1], conv1, sess, file='../results/visualize_40_conv1.png')
    outputFeatureMap(X_new_proc[ii : ii + 1], conv2, sess, file='../results/visualize_40_conv2.png')


