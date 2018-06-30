# Behaviorial Cloning Project

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works. The preprocess.py file contains steps for ingesting the data before performing inference or generating additional data in the training phase.


[//]: # (Image References)

[image1]: ./examples/nvidia.png "NVIDIA Model"
[image2]: ./examples/original.png "Original image from simulator"

---

Running this command will create a new `conda` environment that is provisioned with all needed libraries.
```sh
conda env create -f environment.yml
```

To set up the environment
```sh
source activate carnd-term1
```

Training the model
```sh
python model.py
```

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

Saving images from the run
```sh
python drive.py model.h5 run1
```

Creating a video from images
```sh
python video.py run1 --fps 48
```

The video will run at 48 FPS. The default FPS is 60.


### Model Architecture and Training Strategy

My model consists of a convolution neural network with 3x3 and 5x5 filter sizes and depths between 24 and 64. The architecture of the model is based on [the NVIDIA model.](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) It uses a combination of convolutional and dense layers implemented in the Keras framework.

![alt text][image1]

The model contains a dropout layer in order to reduce overfitting. 

I augmented the training data using the following techniques:
* Incorporating left and right images and adjusting the steering angle as necessary
* Applying shadow to the images
* Adjusting the brightness
* I also created a method called modify_image (model.py 33) that flipped images and translated them by some amount

Original image
![alt text][image2]

I used Adam optimizer with a lower learning rate (0.0001) than default. Since I trained the model on CPU I used a smaller batch size (40). I tried several different epochs until my test error stopped decreasing (10). Training data was chosen to keep the vehicle driving on the road. For collecting the training data I drove the simulator around the track 2-3 times. For training the model I applied several image modifications listed above and used images from the left and right camera.

The overall strategy for deriving a model architecture was to take the NVIDIA model and apply some slight modifications to it. For preprocessing the data, I crop the image to focus on the road and normalize. I convert the image from RGB to YUV as described in the NVIDIA paper. I added a 50% dropout layer between the convolutional and fully-connected layers. In addition I used the ELU activation function to reach convergence faster. In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. To combat the overfitting, I used several image modification techniques to augment the training data. At first I trained the model for only a couple epochs and it quickly went off the track. By increasing the number of epochs both my validation and test error decreased and the vehicle was able to properly complete the course.

I used a 80/20 train test split and used the mean squared error loss function to measure the model's performance. 

#### Final Model Architecture

* Image normalization
* Convolution: 5x5, filter: 24, strides: 2x2, activation: ELU
* Convolution: 5x5, filter: 36, strides: 2x2, activation: ELU
* Convolution: 5x5, filter: 48, strides: 2x2, activation: ELU
* Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
* Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
* Drop out (0.5)
* Fully connected: neurons: 100, activation: ELU
* Fully connected: neurons: 50, activation: ELU
* Fully connected: neurons: 10, activation: ELU
* Fully connected: neurons: 1 (output)

