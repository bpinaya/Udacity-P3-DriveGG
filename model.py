import os
import random
import pandas as pd
import numpy as np
import cv2

import math
import tensorflow as tf

from keras.models import Sequential, Model
from keras.layers import Input, ELU, Convolution2D, MaxPooling2D, Flatten
from keras.layers.core import Dense, Dropout, Activation,Lambda
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras import initializations
from keras.applications.vgg16 import VGG16
from pathlib import Path
my_data = pd.read_csv(os.path.join('data2','driving_log.csv'),index_col = False)
my_data.columns = ['center','left','right','steering','throttle','brake','speed']
udacity_data = pd.read_csv(os.path.join('data','driving_log.csv'))

def change_brightness(image):
    hsv_image = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    brightness = 0.20 + np.random.uniform()
    hsv_image[:,:,2] = hsv_image[:,:,2]* brightness
    return cv2.cvtColor(hsv_image,cv2.COLOR_HSV2RGB)

X_RANGE = 80
Y_RANGE = 30
ANGLE_RANGE = .3
WIDTH = 64
HEIGHT = 64
CHANNELS = 3
BATCH_SIZE = 256
IMAGE_SIZE = (WIDTH,HEIGHT,CHANNELS)
OFF_CENTER_IMG = .25

def x_y_translation(image,angle):
    x_translation = (X_RANGE * np.random.uniform()) - (X_RANGE * 0.5)
    y_translation = (Y_RANGE * np.random.uniform()) - (Y_RANGE * 0.5)
    # Translation Matrix
    # http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html
    # M is the translation Matrix
    # M = np.float32([[1,0,X],[0,1,Y]])
    M = np.float32([[1,0,x_translation],[0,1,y_translation]])
    # Modify the angle for x, given input
    rows,cols,channels = image.shape
    translated_image = cv2.warpAffine(image,M,(cols,rows))
    new_angle = angle + ((x_translation/X_RANGE)*2)*ANGLE_RANGE
    return translated_image,new_angle

def crop_and_resize(image):
    cropped_image = image[50:145,:,:]
    # cv2.INTER_AREA for shrinking and cv2.INTER_CUBIC & cv2.INTER_LINEAR for zooming
    resized_image = cv2.resize(cropped_image,(64,64),interpolation=cv2.INTER_AREA)
    #height, width = img.shape[:2]
    resized_image = np.array(resized_image)
    return resized_image
    #return np.resize(resized_image,(1,64,64,3))

    
def data_augmentation(img_path, angle, threshold, bias):
    
    if (abs(new_angle) + bias) < threshold or abs(new_angle) > 1.:
        return None, None

    image = cv2.imread(img_path)  
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image = change_brightness(image) 
    image, angle = x_y_translation(image, angle)  
    if np.random.randint(2) == 0: 
        img = np.fliplr(image)
        new_angle = -new_angle
    image = crop_and_resize(image)

    return image, new_angle



def get_nvidia_model():
    # End to End Learning for Self-Driving Cars
    # Mariusz Bojarski, Davide Del Testa, Daniel Dworakowski, Bernhard Firner
    # Beat Flepp, Prasoon Goyal, Lawrence D. Jackel, Mathew Monfort, Urs Muller
    # Jiakai Zhang, Xin Zhang, Jake Zhao, Karol Zieba
    # https://arxiv.org/abs/1604.07316
    
    return 0

def get_vgg_model(input_shape):
    # Very Deep Convolutional Networks for Large-Scale Image Recognition
    # Karen Simonyan, Andrew Zisserman
    # https://arxiv.org/abs/1409.1556
    # --------------------------------------------------------------------------
    # RELUs changed with ELUs
    # FAST AND ACCURATE DEEP NETWORK LEARNING BY EXPONENTIAL LINEAR UNITS (ELUS)
    # Djork-Arne Clevert, Thomas Unterthiner & Sepp Hochreiter
    # https://arxiv.org/pdf/1511.07289v1.pdf
    # https://keras.io/layers/advanced-activations/
    input_layer = Input(shape=input_shape)
    input_layer = Lambda(lambda x: x/127.5-.5)(input_layer)
    #input_layer = Convolution2D(3,1,1,border_mode='same',name='input_conv')(input_layer)
    vgg_16_model = VGG16(weights='imagenet', include_top=False, input_tensor=input_layer)
    print(len(vgg_16_model.layers))
    output_layer = vgg_16_model.output
    output_layer = Flatten()(output_layer)
    output_layer = Dense(1024, activation='elu', name='fc1')(output_layer)
    output_layer = Dropout(0.5, name='fc1_dropout')(output_layer)
    
    output_layer = Dense(512, activation='elu', name='fc2')(output_layer)
    output_layer = Dropout(0.5, name='fc2_dropout')(output_layer)
    
    output_layer = Dense(256, activation='elu', name='fc3')(output_layer)
    output_layer = Dropout(0.5, name='fc3_dropout')(output_layer)
    
    output_layer = Dense(128, activation='elu', name='fc4')(output_layer)
    output_layer = Dropout(0.5,name='fc4_dropout')(output_layer)
    
    output_layer = Dense(64, activation='elu', name='fc5')(output_layer)
    output_layer = Dropout(0.5,name='fc5_dropout')(output_layer)
    
    output_layer = Dense(32, activation='elu', name='fc6')(output_layer)
    output_layer = Dropout(0.5,name='fc6_dropout')(output_layer)
    
    output_layer = Dense(1, init='zero', name='output_layer')(output_layer)
    model = Model(input=vgg_16_model.input, output=output_layer)
    
    return model



#
def train_model(model,train_data, validate_data):
    print(len(model.layers))
    model.compile(optimizer=Adam(1e-5),loss='mse')
    val_loss = model.evaluate_generator(validate_data_generator(validate_data),val_samples=128)
    print(val_loss)
    #test_predictions(model,train_data)
    num_runs = 0
    best_value = 999999
    index_best = 0
    while True:
        bias = 1./(num_runs+1.)
        print(num_runs+1,bias)
        history = model.fit_generator(generator=train_data_generator(train_data,bias),
                                     samples_per_epoch=160*128,
                                     nb_epoch=6,
                                     validation_data=validate_data_generator(validate_data),
                                     nb_val_samples=128,
                                     verbose=1)
        #
        num_runs = num_runs +1
        print('wasabi')
        val_loss = history.history['val_loss'][0] 
        if (val_loss < best_value):
            index_best = num_runs
            best_value = val_loss
            save_best_model(model)
        test_predictions(model,train_data)
        #
        if num_runs > 10:
            break
    return best_value, index_best


def train_data_generator(train_data, bias):
    images = np.zeros((BATCH_SIZE, 64, 64, 3), dtype=np.float)
    angles = np.zeros(BATCH_SIZE, dtype=np.float)
    out_idx = 0
    while 1:
        
        index = np.random.randint(len(df))
        angle = df.steering.iloc[idx]

        img_choice = np.random.randint(3)

        if img_choice == 0:
            img_path = 'data/'+train_data.left.iloc[index].strip()
            angle += OFF_CENTER_IMG
        elif img_choice == 1:
            img_path = 'data/'+train_data.center.iloc[index].strip()
        else:
            img_path = 'data/'+train_data.right.iloc[index].strip()
            angle -= OFF_CENTER_IMG

        threshold = np.random.uniform()
        img, angle = data_augmentation(img_path, angle, threshold, bias)

        if img is not None:
            images[out_idx] = img
            angles[out_idx] = angle
            out_idx += 1

        if out_idx >= BATCH_SIZE:
            yield images, angles

            images = np.zeros((BATCH_SIZE, 64, 64, 3), dtype=np.float)
            angles = np.zeros(BATCH_SIZE, dtype=np.float)
            out_idx = 0
            
def validate_data_generator(validate_data):
    while 1:
        x = np.zeros((BATCH_SIZE, 64, 64, 3), dtype=np.float)
        y = np.zeros(BATCH_SIZE, dtype=np.float)

        for index in np.arange(BATCH_SIZE):
            temp = validate_data.center.iloc[index].strip()
            image = cv2.imread('data/'+temp)
            image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            image = crop_and_resize(image)
            x[index] = image
            y[index] = validate_data.steering.iloc[index]

        yield x, y
#

def test_predictions(model,validate_data,number_tests=10):
    for i in range(number_tests):
        index = np.random.randint(len(validate_data))
        temp = udacity_data.center.iloc[index].strip()
        image = cv2.imread('data/'+temp)
        image = crop_and_resize(image)
        real_angle = validate_data.steering.iloc[index]
        image = image[None, :, :, :]
        predicted_angle = model.predict(image,batch_size=1)
        print('Prediction: '+str(i))
        print(real_angle,predicted_angle[0][0])
#
def save_best_model(model):
    if Path('model.json').is_file():
        os.remove('model.json')
        print('Model already there')
    if Path('model.h5').is_file():
        os.remove('model.h5')
    json_string = model.to_json()
    with open('model.json','w') as outfile:
        outfile.write(json_string)
    model.save_weights('model.h5')
    
def main():
    my_data = pd.read_csv(os.path.join('data2','driving_log.csv'),index_col = False)
    my_data.columns = ['center','left','right','steering','throttle','brake','speed']
    udacity_data = pd.read_csv(os.path.join('data','driving_log.csv'))
    # Shuffle images with .sample(frac=1) and take 20% to validate
    validate_ud, train_ud = np.split(udacity_data.sample(frac=1),[int(len(udacity_data)*.2)])
    validate_my, train_my = np.split(my_data.sample(frac=1),[int(len(my_data)*.2)])
    del my_data, udacity_data
    #print(len(validate_ud)) #1607
    #print(len(train_ud)) #6429
    #nvidia_steering_model = get_nvidia_model()
    vgg_steering_model = get_vgg_model(IMAGE_SIZE)
    test_predictions(vgg_steering_model,validate_ud)
    best_value,index_best = train_model(vgg_steering_model,validate_ud,train_ud)
    print('FINAL RESULTS')
    print(best_value,index_best)
    test_predictions(vgg_steering_model,validate_ud)

if __name__ == "__main__":
    main()
