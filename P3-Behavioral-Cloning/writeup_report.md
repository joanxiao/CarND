
## Behavioral Cloning Project

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md summarizing the results

#### 2. Submssion includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submssion code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model arcthiecture has been employed

My model consists of a convolution neural network, with 5 convolutional layers of filters of size 5x5 and 3x3 and depths between 24 and 64, and 3 fully connected layers (model.py lines 207-227)

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras Lambda layer (code line 208).

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting.

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 246-249). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer with default learning rate of 0.001 (model.py line 229). I've tested with lower learning rates but it seems the default works best.

#### 4. Appropriate training data

I used the sample Udacity training data provided by the project.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to implement a neural network based on the [Nvidia paper](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf). The authors of the paper trained a CNN to map raw pixels from a single
front-facing camera directly to steering commands and the car was able to drive autonomously.

My first step was to build a convolution neural network model similar to the Nvidia paper due to the similarity of the problems. And I used the left and right camera images as well with the steering adjusted.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

To combat the overfitting, I added dropout at various layers of the model and the validation MSE dropped along wtih training MSE as the training goes.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, such as when it needs to make a right turn. To improve the driving behavior in these cases, I did more augmentation on the data such as randomly translating images, flipping the images and changing the brightness of the iamges by a random value.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

However, the car was not able to complete track 2. I did more data augmentation by generating a random shadow of polygon shape on top of the images, and that helped the car complete track 2 as well.

#### 2. Final Model Architecture

The final model architecture (model.py lines 207-227) consisted of a convolution neural network with the following layers and layer sizes:

<pre>
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
Normalize (Lambda)               (None, 160, 320, 3)   0           lambda_input_1[0][0]             
____________________________________________________________________________________________________
Cropping (Cropping2D)            (None, 80, 320, 3)    0           Normalize[0][0]                  
____________________________________________________________________________________________________
Resize (Lambda)                  (None, 66, 200, 3)    0           Cropping[0][0]                   
____________________________________________________________________________________________________
Conv1 (Conv2D 24@5x5)            (None, 31, 98, 24)    1824        Resize[0][0]                     
____________________________________________________________________________________________________
Conv2 (Conv2D 36@5x5)            (None, 14, 47, 36)    21636       Conv1[0][0]                      
____________________________________________________________________________________________________
Dropout1 (Dropout)               (None, 14, 47, 36)    0           Conv2[0][0]                      
____________________________________________________________________________________________________
Conv3 (Conv2D 48@5x5)            (None, 5, 22, 48)     43248       Dropout1[0][0]                   
____________________________________________________________________________________________________
Dropout2 (Dropout)               (None, 5, 22, 48)     0           Conv3[0][0]                      
____________________________________________________________________________________________________
Conv4 (Conv2D 64@3x3)            (None, 3, 20, 64)     27712       Dropout2[0][0]                   
____________________________________________________________________________________________________
Conv5 (Conv2D 64@3x3)            (None, 1, 18, 64)     36928       Conv4[0][0]                      
____________________________________________________________________________________________________
Dropout3 (Dropout)               (None, 1, 18, 64)     0           Conv5[0][0]                      
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 1152)          0           Dropout3[0][0]                   
____________________________________________________________________________________________________
FC1 (Dense)                      (None, 100)           115300      flatten_1[0][0]                  
____________________________________________________________________________________________________
Dropout4 (Dropout)               (None, 100)           0           FC1[0][0]                        
____________________________________________________________________________________________________
FC2 (Dense)                      (None, 50)            5050        Dropout4[0][0]                   
____________________________________________________________________________________________________
Dropout5 (Dropout)               (None, 50)            0           FC2[0][0]                        
____________________________________________________________________________________________________
FC3 (Dense)                      (None, 10)            510         Dropout5[0][0]                   
____________________________________________________________________________________________________
Output (Dense)                   (None, 1)             11          FC3[0][0]                        
====================================================================================================

</pre>

Here is a visualization of the architecture:

![Network architecuture](https://github.com/joanxiao/SDCND-Behavioral-Cloning-P3/blob/master/model.png)

#### 3. Creation of the Training Set & Training Process

I initially tried to run the simulator to create my own training data, but I found it very difficult. I had great difficulties to use the beta version and the mouse, so I settled on the regular version of the simulator with a keyboard. Although I was able to complete the laps, using a keyboard does not control the steering very well and I worried that I might run into problem with creating recovery data later on.

Fortunately Udacity provided sample data so I decided to use that and see how it goes.

I performed some data exploration, and found that over 50% of the center images have 0 steering angle, so the dataset is very imbalanced. Moreover, there are more positve steering angles than negative ones. Below is the histogram of the steering angles.

![Histogram](https://github.com/joanxiao/SDCND-Behavioral-Cloning-P3/blob/master/images/hist_before.jpg)

To balance the data, I dropped a random 75% of these rows with 0 steering angle. I then randomly shuffled the data set and put 20% of the data into a validation set. However, later during training I found that the training loss decreased at very slow rate. So I changed the splitting to use 10% as the validation data, and I was able to obtain both the training  and validation loss at 0.02 after about 3 epochs.

I loaded the images from center, left and right cameras. For the left camera images, I added the steering angle from center camera by 0.2, and for the right camera image, I subtracted that steering angle by 0.2. I then flipped the images so that we have balanced data with left and right steering angles. Other augmentations include horizontally shifting the images and casting a shandow of random polygon shape on the images. 

Below are 25 random images from the training set before the augmentation, with the steering angle shown above each image:

![Random training images](https://github.com/joanxiao/SDCND-Behavioral-Cloning-P3/blob/master/images/random_images.jpg)

Below are 5 random images from the training set before the augmentation, followed by 4 augmented images in the same row. Steering angles ar shown above the images.

![Random augmented images](https://github.com/joanxiao/SDCND-Behavioral-Cloning-P3/blob/master/images/augmented_images.jpg)

For all images, I cropped the top 55 and bottom 25 pixels, and then resized them to 66x200 per the Nvidia architecture. The validation set helped determine if the model was over or under fitting. I used fit_generator to generate batches of 128 augmented images to train the model, and I used a callback to save the model at the end of each epoch. I trained for 10 models and the best model was obtained at epoch 5. I used an adam optimizer and tried various learning rate, although I settled with the default learning rate of 0.001 at the end.

## Reflections

This project has taken me the longest time to complete among all the projects in this program so far. I stubmled upon several issues with the simulator. My development environment is a Linux VM, and unfortunately I'm unable to run the simulator on the VM. There are other students reporting similar problem but I haven't seen a solution. So I resorted to train the model on linux, and then copy the model to a Windows environment and run the simulator there. Then on Windows I ran into issue with the beta simulator.

But it's been very fulfilling to be able to complete the project after all these obstacles. The car rides beautifully on track 2 currently, but on track 1 some of the turns are still not smooth, even though I've tried to modify the data augmentation and tune the model in various ways. I'm interested in finding out what other improvements I can do to make the car drive more smoothly.
