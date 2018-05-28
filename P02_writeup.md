# **Traffic Sign Recognition** 

## Objective
The objective of this projects are to:
* Analyze and visualize the provided German traffic sign dataset
* Design, train, and test a neural network model to classify the traffic
sign images 
* Use the model to predict traffic sign classes on new images
* Analyze the softmax probabilities of the new images
* Visualize convolution layer output feature maps


[//]: # (Image References)

[image01]: ./results/count_by_dataset.png "Image Count By Dataset"
[image02]: ./results/count_by_traffic_sign.png "Traffic Sign Occurrence"
[image03]: ./results/preprocessing.png "Preprocessing"
[new_images]: ./results/new_images.png "New Images (Original)"
[precision_recall]: ./results/precision_and_recall_test.png "Precision And Recall"
[optional_01]: ./results/visualize_11.png
[optional_02]: ./results/visualize_11_conv1.png
[optional_03]: ./results/visualize_11_conv2.png
[optional_04]: ./results/visualize_40.png
[optional_05]: ./results/visualize_40_conv1.png
[optional_06]: ./results/visualize_40_conv2.png

 
---

## Dataset Exploration

### 1. Basic Dataset Summary

I used the numpy library to calculate summary statistics of the traffic
signs dataset:

* The size of training set is **34,799**
* The size of the validation set is **4,410** 
* The size of test set is **12,630**
* The shape of a traffic sign image is **(32, 32)**
* The number of unique classes/labels in the dataset is **43**

### 2. Visualize Dataset Summary
![Count_By_Dataset][image01]

![Traffic_Sign_Occurrence][image02]


## Design and Test a Model Architecture

### 1. Image Preprocessing

**Grayscaling & Normalization**

First, the traffic sign datasets are converted to grayscale using `cv2.cvtColor()`
function.  Then, each pixel value is normalized to `[-1, 1)` using
`(pixel - 128.0) / 128.0` equation.

While the original color images can directly be used by increasing the
color channel dimension to 3, I decided to convert the images into the
grayscale.  In identifying traffic sign classes, the color does not play
crucial role. By reducing the color channel, the complexity of the neural
networks is reduced without losing the prediction accuracy.

The weights and biases of the neural networks are initialized with a normal
distribution of 0 mean and 0.1 standard deviation.  For this initial condition,
the datasets resembling standard normal distribution (0 mean 1 standard
deviation) works more efficiently for model training. 

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

Adam (Adaptive Momentum Estimation) optimizer is used to train the model.
Adam maintains per-parameter learning rates and adapts learning rates
based on momentum.

Adam typically requires less hyper-parameter tuning and it achieves good
results fast.  This makes Adam a good choice of the optimizers.

Dropout layers are added to avoid overfitting the model to the training set.
The keep probability of 0.4 is selected as it yields a close correlation
between training and validation accuracies while converging efficiently.
 

### 4. Approach to Solution

My final model results were:
* training set accuracy of 99.84%
* validation set accuracy of 95.85%
* test set accuracy of 95.05%

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
However, it slowed down the model conversion and the number of epochs is
increased to compensate the slowdown.  

In order to achieve further accuracy improvement, the sizes of hidden layer
outputs are increased.

Comparing to that the digit has 10 types, there are 43 distinct traffic
signs in the provided dataset.  Therefore, I decided to increase the sizes
of hidden output layers.  This increment resulted in some accuracy
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


### 2. Predictions on New Image Set

| Image	File | Sign Class             | Prediction              |
|:----------:|:----------------------:|:-----------------------:|
| 11.1.jpg   | Right-of-way at the ...| Right-of-way at the ... |
| 13.1.jpg   | Yield                  | Yield                   |
| 22.1.jpg   | Bumpy road             | **Yield**               |
| 25.1.jpg   | Road work              | Road work               |
| 40.1.jpg   | Roundabout mandatory   | Roundabout mandatory    |
| 13.2.png   | Yield                  | Yield                   |
| 22.2.png   | Bumpy road             | Bumpy road              |
| 25.2.png   | Road work              | Road work               |

For 5 original images, the model correctly predicted 4 of them.  This is
80% accuracy and it's worse than test set accuracy of 95.05%.  But, five
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

#### Precision and Recall

The precision and the recall values are computed from the test dataset.
All newly selected five traffic signs have relatively high precision and
recall values.

One _Bumpy road_ sign is identified as _Yield_.  However, the recall of
the _Bumpy road_ (94.16%) and the precision of the _Yield_ (99.86%) are
high.  As suggested above, it is likely that the error on _Bumpy road_
sign prediction is caused by irrelevant details (noise) in the original image.

**Precision & Recall from Test Set for Five New Images**

| Sign Class                  | Precision   | Recall    |
|:---------------------------:|:-----------:|:---------:| 
| 11 - Right-of-way at the ...| 89.25%      | 90.95%    |
| 13 - Yield                  | 99.86%      | 99.58%    |
| 22 - Bumpy road             | 95.76%      | 94.16%    |
| 25 - Road work              | 99.77%      | 92.70%    |
| 40 - Roundabout mandatory   | 94.93%      | 83.33%    |


Below listed are traffic signs with low precision and recall.  The precision
value gets as low as 67.16% on the _Beware of ice/snow_ sign,
and the recall gets 50% recall on the _Pedestrians_ sign.

The graph in the earlier section for traffic sign occurrence in training set
shows that all of these five low fidelity traffic signs have a small number
of training samples.  I believe that adding more samples on these low
fidelity signs could improve overall model accuracy. 

**Traffic Signs with Low Precision or Recall**

| Sign Class                        | Precision   | Recall    |
|:---------------------------------:|:-----------:|:---------:| 
| 24 - Road narrows on the right    | 93.54%      | 64.44%    |
| 27 - Pedestrians                  | 81.08%      | 50.00%    |
| 29 - Bicycles crossing            | 68.99%      | 98.88%    |
| 30 - Beware of ice/snow           | 67.16%      | 60.00%    |
| 41 - End of no passing            | 87.23%      | 68.33%    |


![precision_recall][precision_recall]


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
| Yield                  | 99.33%      |
| Speed limit (50km/h)   | 0.65%       |
| Priority road          | 0.01%       |
| No passing             | 0.00%       |
| Keep left              | 0.00%       |

File Name: **13.2.png**

| Prediction             | Confidence  |
|:----------------------:|:-----------:|
| Yield                  | 100.00%     |
| Ahead only             | 0.00%       |
| Priority road          | 0.00%       |
| Keep right             | 0.00%       |
| Road work              | 0.00%       |

#### Bumpy road

File Name: **22.1.jpg** - _Incorrect Prediction_

| Prediction             | Confidence  |
|:----------------------:|:-----------:|
| Yield                  | 67.58%      |
| Priority road          | 16.93%      |
| Speed limit (30km/h)   | 15.28%      |
| Speed limit (50km/h)   | 0.11%       |
| Stop                   | 0.05%       |

File Name: **22.2png**

| Prediction                | Confidence  |
|:-------------------------:|:-----------:|
| Bumpy road                | 99.99%      |
| Bicycles crossing         | 0.01%       |
| Beware of ice/snow        | 0.00%       |
| Traffic signals           | 0.00%       |
| Keep right                | 0.00%       |

#### Road work

File Name: **25.1.jpg**

| Prediction             | Confidence  |
|:----------------------:|:-----------:|
| Road work              | 47.97%      |
| Double curve           | 39.78%      |
| Wild animals crossing  | 7.61%       |
| Keep left              | 2.35%       |
| Bicycles crossing      | 0.72%       |

File Name: **25.2.png**

| Prediction                    | Confidence  |
|:-----------------------------:|:-----------:|
| Road work                     | 100.00%     |
| Priority road                 | 0.00%       |
| Bicycles crossing             | 0.00%       |
| Keep right                    | 0.00%       |
| Beware of ice/snow            | 0.00%       |

#### Roundabout mandatory

File Name: **40.1.jpg**

| Prediction             | Confidence  |
|:----------------------:|:-----------:|
| Roundabout mandatory   | 100.00%     |
| Priority road          | 0.00%       |
| Speed limit (100km/h)  | 0.00%       |
| No entry               | 0.00%       |
| Speed limit (120km/h)  | 0.00%       |


## Visualize Neural Network's State

### 1. Visualize Convolution Layer Output Feature Map

Two images, _Right-of-way at the next intersection_ and _Roundabout mandatory_,
are used to visualize the feature map of two convolution layers.

The first layer output still resembles the overall shape of the input image.
The resemblance gets more abstracted out in the second layer output.  Both
layer feature maps roughly identify regions relevant to the traffic sign.
In addition, each feature map seems to get activated by different features
of the input image such as parts of sign boundaries, regions of the sign
symbols etc. 

#### Right-of-way at the next intersection

**Grayscale Image**

![optional_01][optional_01]

**Convolution Layer 1**

![optional_02][optional_02]

**Convolution Layer 2**

![optional_03][optional_03]


#### Roundabout mandatory

**Grayscale Image**

![optional_04][optional_04]

**Convolution Layer 1**

![optional_05][optional_05]

**Convolution Layer 2**

![optional_06][optional_06]

