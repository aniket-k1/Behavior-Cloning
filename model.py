import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten
import cv2
import matplotlib.image as mpimg
from preprocess import preprocess
import argparse
import os

IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 66, 200, 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)

np.random.seed(0)

def load_image(data_dir, image_file):
    return mpimg.imread(os.path.join(data_dir, image_file.strip()))

def choose_image(data_dir, center, left, right, steering_angle):
    # Randomly pick which angle of image
    choice = np.random.choice(3)
    if choice == 0:
    	# Modify steering angle to account for different camera
        return load_image(data_dir, left), steering_angle + 0.2
    elif choice == 1:
        return load_image(data_dir, right), steering_angle - 0.2
    return load_image(data_dir, center), steering_angle


def modify_image(image, steering_angle):
    # Mirror the image, change the angle accordingly
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
        steering_angle = -steering_angle

    # Translate the image a random amount
    trans_x = 100 * (np.random.rand() - 0.5)
    trans_y = 10 * (np.random.rand() - 0.5)
    steering_angle += trans_x * 0.002
    trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    height, width = image.shape[:2]
    image = cv2.warpAffine(image, trans_m, (width, height))
    return image, steering_angle


def change_shadow(image):
	# Darken the left or right side by random amount
    height, width = image.shape[0:2]
    mid = np.random.randint(0, width)
    factor = np.random.uniform(0.6,0.8)
    if np.random.rand() > .5:
        image[:,0:mid,0] *= factor
    else:
        image[:,mid:width,0] *= factor
    return image


def change_brightness(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
    hsv[:,:,2] =  hsv[:,:,2] * ratio
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def generate_additional_data(data_dir, center, left, right, steering_angle):
    image, steering_angle = choose_image(data_dir, center, left, right, steering_angle)

    image, steering_angle = modify_image(image, steering_angle)
    
	image = change_shadow(image)
	image = change_brightness(image)
    return image, steering_angle


def batch_generator(data_dir, image_paths, steering_angles, batch_size, is_training):
    images = np.empty([batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
    steers = np.empty(batch_size)
    while True:
        i = 0
        for index in np.random.permutation(image_paths.shape[0]):
            center, left, right = image_paths[index]
            steering_angle = steering_angles[index]
            # argumentation
            if is_training and np.random.rand() < 0.6:
                image, steering_angle = generate_additional_data(data_dir, center, left, right, steering_angle)
            else:
                image = load_image(data_dir, center) 
            # add the image and steering angle to the batch
            images[i] = preprocess(image)
            steers[i] = steering_angle
            i += 1
            if i == batch_size:
                break
        yield images, steers

def load_data():
	data_df = pd.read_csv(os.path.join('data', 'driving_log.csv'), names=['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed'])	
	X = data_df[['center', 'left', 'right']].values
	y = data_df['steering'].values
	X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=0)
	return X_train, X_valid, y_train, y_valid

def build_model():
	model = Sequential()

	# Regularization
	model.add(Lambda(lambda x:x/127.5-1.0, input_shape=INPUT_SHAPE))

	model.add(Conv2D(24, 5, 5, activation='elu', subsample=(2, 2)))
	model.add(Conv2D(36, 5, 5, activation='elu', subsample=(2, 2)))
	model.add(Conv2D(48, 5, 5, activation='elu', subsample=(2, 2)))
	model.add(Conv2D(64, 3, 3, activation='elu'))
	model.add(Conv2D(64, 3, 3, activation='elu'))

	model.add(Dropout(0.5))
	model.add(Flatten())

	model.add(Dense(100, activation='elu'))
	model.add(Dense(50, activation='elu'))
	model.add(Dense(10, activation='elu'))
	model.add(Dense(1))
	
	model.summary()

	return model


def train_model(model, X_train, X_valid, y_train, y_valid):
	checkpoint = ModelCheckpoint('model-{epoch:03d}.h5',
									monitor='val_loss',
									verbose=0,
									save_best_only=True,
									mode='auto')
	
	# 0.0001 instead of .001
	model.compile(loss='mean_squared_error', optimizer=Adam(lr=1.0e-4))

    batch_size = 40
    samples_per_epoch = 20000
    epochs = 10
	model.fit_generator(batch_generator('data', X_train, y_train, batch_size, True),
											samples_per_epoch,
											epochs,
											max_q_size=1,
											validation_data=batch_generator('data', X_valid, y_valid, batch_size, False),
											nb_val_samples=len(X_valid),
											callbacks=[checkpoint],
											verbose=1)


def main():
    data = load_data()
    model = build_model()
    train_model(model, *data)

if __name__ == '__main__':
    main()

