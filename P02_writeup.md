# **Traffic Sign Recognition** 

## Objective
The objective of this projects are to:
* Analyze and visualize the provided German traffic sign data set
* Design, train, and test a neural network model to classify the traffic
sign images 
* Use the model to predict traffic sign classes on new images
* Analyze the softmax probabilities of the new images


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"
[image01]: ./results/count_by_dataset.png "Image Count By Dataset"
[image02]: ./results/count_by_traffic_sign.png "Traffic Sign Occurrence"
[image03]: ./results/preprocessing.png "Preprocessing"
[new_images]: ./results/new_images.png "New Images (Original)"
 

---

## Dataset Exploration

### 1. Basic Dataset Summary

I used the numpylibrary to calculate summary statistics of the traffic
signs data set:

* The size of training set is **34,799**
* The size of the validation set is **4,410** 
* The size of test set is **12,630**
* The shape of a traffic sign image is **(32, 32)**
* The number of unique classes/labels in the data set is **43**

### 2. Visualize Dataset Summary
![Count_By_Dataset][image01]

![Traffic_Sign_Occurrence][image02]


## Design and Test a Model Architecture

### 1. Image Preprocessing

**Grayscaling & Normalization**

First, the traffic sign datasets are converted to grayscale using `cv2.cvtColor()`
function.  Then, each pixel value is normalized to `[-1, 1)` using
`(pixel - 128.0) / 128.0` equation.

While the original color images can be used by increasing the color channel
dimension to 3, I decided to convert the images into the grayscale.  In
identifying traffic sign classes, the color does not play crucial role.
By reducing the color channel, the complexity of the neural networks is
reduced without losing the prediction accuracy.

The weights and biases of the neural networks are initialized with a normal
distribution of 0 mean and 0.1 standard deviation.  For this initial condition,
the datasets resembling standard normal distribution (0 mean 1 standard
deviation) works more efficiently during model training. 

The plot below demonstrates the traffic sign images at each preprocessing step.
As one can see, the normalization does not change the visual representation
of the image. 

![Grayscale_Normalization][image03]


### 2. Neural Networks Model Architecture

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   					| 
| Convolution 5x5     	| 1x1 stride, VALID padding, outputs 28x28x12 	|
| ReLU					|												|
| Max pooling	      	| 2x2 kernel and stride,  outputs 14x14x12 		|
| Convolution 5x5	    | 1x1 stride, VALID padding, outputs 10x10x32	|
| ReLU					|												|
| Max pooling	      	| 2x2 kernel and stride,  outputs 5x5x32 		|
| Flattening    		| 5x5x32 => (800)        						|
| Fully connected		| (800) => (240)								|
| ReLU					|												|
| Dropout				| keep_rate: 0.4 for training, 1.0 for testing*	|
| Fully connected		| (240) => (120)    							|
| ReLU					|												|
| Dropout				| keep_rate: 0.4 for training, 1.0 for testing	|
| Fully connected		| (120) => (43)									|
| Softmax       		|               								|
| Cross Entropy    		|               								|

testing* - dropout keep_rate is 1.0 for both testing and validation


### 3. Model Training

| Parameter             | Setting           |
|:---------------------:|:-----------------:|
| Batch Size            | 128               |
| Number of Epochs      | 20                |
| Learning Rate         | 0.001             |
| Dropout keep_prob     | 0.4               |
| Optimizer             | AdamOptimizer     |

Adam (Adaptive Momentum Estimation) optimizer is used in order to train
the model. Adam maintains per-parameter learning rates and adapts learning
rates based on momentum.

Adam typically requires less hyper-parameter tuning and it achieves good
results fast.  This makes Adam a good choice of the optimizers.

Dropout layers are added to avoid overfitting the model to the training set.
The keep probability of 0.4 is selected as it yields a close correlation
between training and validation accuracies while converging efficiently.
 

### 4. Approach to Solution

My final model results were:
* training set accuracy of 99.78%
* validation set accuracy of 97.05%
* test set accuracy of 94.84%

My model is first based on the LeNet digit classifier by modifying its output
class dimension.  The LeNet architecture was chosen for the base model
because of its Convolutional Networks based approach as a general vision
problem.

While the base model achieved high training accuracy relatively quickly
(train accuracy of over 99% in 10 epochs), it experienced the overfitting
issue.  The model underperformed on the validation set by the accuracy of
almost 10%.

In order to address the overfitting problem, the dropout layer is introduced
after the ReLU layer of the two hidden fully connected layer.  The dropout
layer reduced the gap between the training and validation accuracies.
However, it slowed down the model training and the number of epochs is
increased to compensate the slow down.  

In order to achieve further accuracy improvement, the sizes of hidden layer
outputs are increased.

Comparing to that the digit has 10 types, there are 43 distinct traffic
signs in the provided dataset.  Therefore, I decided to increase the size
of hidden output layer.  This increment resulted in some accuracy
enhancement.

The accuracy calculations and results of the training and validation
dataset, and the test data are located in the 7th and 8th cell of Ipython
notebook respectively.


## Test a Model on New Images

### 1. New Traffic Sign Images

Here are German traffic signs that I found on the web:

![New Images][new_images]

The new image set contains five original images and three additional
processed images.  The five original images contain features that make
the sign classification difficult such as not centered, non-squared shape,
background noises, rotations, and reflections.

#### Five Original Images

| Image	File | Sign Class                  | Comments                    |
|:----------:|:---------------------------:|:---------------------------:| 
| 11.1.jpg   | 11 - Right-of-way at the ...|                             |
| 13.1.jpg   | 13 - Yield                  | Not centered                |
| 22.1.jpg   | 22 - Bumpy road             | Rotated. Background noises  |
| 25.1.jpg   | 25 - Road work              | Reflections                 |
| 40.1.jpg   | 40 - Roundabout mandatory   |                             |

#### Three Processed Images

| Image	File | Sign Class                  | Comments                    |
|:----------:|:---------------------------:|:---------------------------:| 
| 13.2.png   | 13 - Yield                  | Cropped to center           |
| 22.2.png   | 22 - Bumpy road             | Cropped to center           |
| 25.2.png   | 25 - Road work              | Cropped to center           |

The five sign images contain some features that challenges


### 2. Predictions on New Image Set

| Image	File | Sign Class             | Prediction              |
|:----------:|:----------------------:|:-----------------------:|
| 11.1.jpg   | Right-of-way at the ...| Right-of-way at the ... |
| 13.1.jpg   | Yield                  | Yield                   |
| 22.1.jpg   | Bumpy road             | **Speed limit (20km/h)**|
| 25.1.jpg   | Road work              | Road work               |
| 40.1.jpg   | Roundabout mandatory   | Roundabout mandatory    |
| 13.2.png   | Yield                  | Yield                   |
| 22.2.png   | Bumpy road             | Bumpy road              |
| 25.2.png   | Road work              | Road work               |

For 5 original images, the model correctly predicted 4 of them.  This is
80% accuracy and it's worse than test set accuracy of 94.84%.  But, five
new image samples are small and it's hard to draw conclusive analysis
based on this.

For 3 additional processed images, the model was able to predicted all 3 of
them correctly.  The original versions of these three images contains
relatively large horizontal margins.  The large margin makes the area of
interest out of the center, and it also introduces noises from the
background areas.  Therefore, it is natural that the prediction accuracy
increases when images are cropped around the area of interest.

This analysis highlights that it is crucial to prepare sign images with
less margin areas for better predictions.  This result also suggests that
adding augmented images (especially zoom out and shift) into train sets
would make the model more robust and less susceptible to crude sign image
preparations. 

The total new image accuracy is 87.50% considering both the original and
processed images.



**TODO: _Precision and Recall_**



### 3. Prediction Confidence - Softmax Probability

#### Right-of-way at the next intersection

File Name: **11.1.jpg**

| Prediction             | Confidence  |
|:----------------------:|:-----------:|
| Right-of-way at the ...| 100.00%     |
| Beware of ice/snow     | 0.00%       |
| Double curve           | 0.00%       |
| Pedestrian             | 0.00%       |
| Roundabout mandatory   | 0.00%       |

#### Yield

File Name: **13.1.jpg**

| Prediction             | Confidence  |
|:----------------------:|:-----------:|
| Yield                  | 100.00%     |
| Priority road          | 0.00%       |
| No vehicles            | 0.00%       |
| Ahead only             | 0.00%       |
| No passing             | 0.00%       |

File Name: **13.2.png**

| Prediction             | Confidence  |
|:----------------------:|:-----------:|
| Yield                  | 100.00%     |
| Priority road          | 0.00%       |
| Ahead only             | 0.00%       |
| Keep left              | 0.00%       |
| Road work              | 0.00%       |

#### Bumpy road

File Name: **22.1.jpg** - _Incorrect Prediction_

| Prediction             | Confidence  |
|:----------------------:|:-----------:|
| Speed limit (30km/h)   | 93.59%      |
| Turn left ahead        | 2.49%       |
| Speed limit (20km/h)   | 2.03%       |
| Turn right ahead       | 1.70%       |
| General caution        | 0.08%       |

File Name: **22.2png**

| Prediction                | Confidence  |
|:-------------------------:|:-----------:|
| Bumpy road                | 99.76%      |
| Bicycles crossing         | 0.15%       |
| Road work                 | 0.10%       |
| Road narrows on the right | 0.00%       |
| Beware of ice/snow        | 0.00%       |

#### Road work

File Name: **25.1.jpg**

| Prediction             | Confidence  |
|:----------------------:|:-----------:|
| Road work              | 99.99%      |
| Bumpy road             | 0.01%       |
| Bicycles crossing      | 0.00%       |
| Beware of ice/snow     | 0.00%       |
| Yield                  | 0.00%       |

File Name: **25.2.png**

| Prediction                    | Confidence  |
|:-----------------------------:|:-----------:|
| Road work                     | 100.00%     |
| Dangerous curve to the right  | 0.00%       |
| Yield                         | 0.00%       |
| Bumpy road                    | 0.00%       |
| Beware of ice/snow            | 0.00%       |

#### Roundabout mandatory

File Name: **40.1.jpg**

| Prediction             | Confidence  |
|:----------------------:|:-----------:|
| Roundabout mandatory   | 99.83%      |
| Speed limit (100km/h)  | 0.17%       |
| Priority road          | 0.00%       |
| Speed limit (120km/h)  | 0.00%       |
| Speed limit (80km/h)   | 0.00%       |


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


