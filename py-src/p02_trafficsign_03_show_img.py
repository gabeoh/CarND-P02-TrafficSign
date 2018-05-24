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
n_validation = X_valid.shape[0]

# Number of testing examples.
n_test = X_test.shape[0]

# What's the shape of an traffic sign image?
image_shape = X_train.shape[1:3]

# How many unique classes/labels there are in the dataset.
n_classes = np.unique(np.append(np.append(y_train, y_valid), y_test)).shape[0]

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)


#%% Step 1-0 - Read signnames.csv
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
### Feel free to use as many code cells as needed.
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
img_indexes = get_img_indexes_for(count_train, display_count, 24)
print(img_indexes)
img_indexes = img_indexes[0:display_count]
for i in range(display_count):
    plt.subplot(img_row, img_col, i + 1)
    img_index = img_indexes[i]
    img_label = y_train[img_index]
    plt.title('I:{}, L:{}\n{}'.format(img_index, img_label, signnames[img_label]))
    plt.imshow(X_train[img_index])
plt.show()


