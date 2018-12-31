import numpy as np
import pandas as pd

from keras.models import Model, Sequential
from keras.layers import  GlobalMaxPooling2D, Dense, Dropout, BatchNormalization, Flatten
from keras.optimizers import Adam, Adadelta
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.utils import to_categorical
from keras.applications.nasnet import NASNetLarge

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import os
from os import path, listdir
import gc

import cv2
# from PIL import Image


# Getting The Data
train_dset = pd.read_csv('./data/train.csv')
train_image_path = 'C:/Users/abtex/Desktop/100DaysOfMLCode/kaggleKernels/humpback whale/data/train'
test_image_path = 'C:/Users/abtex/Desktop/100DaysOfMLCode/kaggleKernels/humpback whale/data/test'

test_csv = pd.read_csv('./data/sample_submission.csv')


train_y = train_dset['Id'].values
labels = pd.unique(train_y)
train_y = pd.Series(train_y).astype('category').cat.codes
idx_to_label = dict(zip(np.unique(train_y),labels))

train_x = []
test_x  = []
train_imgs = train_dset['Image'].values

data_gen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=10,
        zoom_range = 0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=False,
        vertical_flip=False
		)

img_dim = (128, 128)
num_channels = 3
num_classes = len(np.unique(train_y))


for img in train_imgs:
	image = cv2.imread(path.join(train_image_path, img))
	image = cv2.resize(image, img_dim)
	image = np.array(image)
	train_x.append(image)

for img in listdir(test_image_path):
	image = cv2.imread(path.join(test_image_path, img))
	image = cv2.resize(image, img_dim)
	image = np.array(image)
	test_x.append(image)

train_x = np.array(train_x)
test_x = np.array(test_x)

print(train_x.shape)
print(test_x.shape)

base_model = NASNetLarge(
					input_shape=(*img_dim, num_channels,),
					include_top=False,
					weights='imagenet',
					input_tensor=None,
					pooling=None,
				)

for layer in base_model.layers:
	layer.trainable = False

x = base_model.output
x = BatchNormalization()(x)
x = Flatten()(x)
x = Dense(4096, activation='relu')(x)
x = Dropout(0.3)(x)
x = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.inputs, outputs=x)

model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-3))

# data_gen.fit(train_x)

reduceLR = ReduceLROnPlateau(
							monitor='val_acc', 
							patience=2, 
							verbose=1, 
							factor=0.2, 
							min_lr=0.00001
						)

earlystop = EarlyStopping(monitor='acc', min_delta=0, patience=1, verbose=0, mode='auto', baseline=None)

tensorboard = TensorBoard(log_dir='./logs', write_images=True, batch_size=128, write_graph=True, write_grads=True)

checkpoint = ModelCheckpoint('./checkpoints/', monitor='val_loss', save_best_only=True)

# model.fit_generator(
# 					data_gen.flow(train_x, train_y, batch_size=128),
# 					shuffle=True, 
# 					epochs=15,
# 					callbacks=[earlystop, tensorboard, reduceLR, checkpoint]
# 				)

model.fit(train_x, train_y, batch_size=128, epochs=10, callbacks=[earlystop, tensorboard, reduceLR, checkpoint])

def topK(predictions, k = 5):
	predictions = [np.absolute(np.argsort(-1*x))[:k] for x in predictions]
	predictions = [idx_to_label[i] for prediction in predictions for i in prediction]
	return predictions

preds = model.predict(test_x)
prediction = topK(preds, 5)

print(preds[1])
print(prediction[1])

predictions = pd.DataFrame(data=test_csv)
predictions.drop(columns=['Id'], inplace=True)
predictions['Id'] = prediction.join(' ')
predictions.to_csv('submission.csv', index=False)