
# coding: utf-8
import os
import pandas as pd
import numpy as np
from numpy import random
import cv2
import time


from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Activation, Flatten, Merge, merge, Lambda
from keras.layers import Embedding
from keras.layers.normalization import BatchNormalization
from keras.layers import Convolution2D, Cropping2D
from keras.utils.np_utils import to_categorical
from keras.models import Model
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import Callback
from keras.utils.visualize_util import plot
import gc
from keras import backend 

INPUT_SHAPE = (160, 320, 3)
BATCH_SIZE = 128
EPOCHS = 10
SAMPLES_PER_EPOCH = 200 * BATCH_SIZE

def load_data(data_file):
    start = time.time()
    print('loading data from ' + data_file_path)
    df = pd.read_csv(data_file)
    print('found {} rows in file'.format(len(df)))
    
    #keep only 25% of the rows with 9 steering angle
    df = df.drop(df.query('steering==0').sample(frac=0.75).index)    
    df = df.sample(frac=1).reset_index(drop=True)

    df_train, df_val = train_test_split(df, test_size=0.1)
    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    print('training data size: {}'.format(len(df_train)))
    print('validation data size: {}'.format(len(df_val)))
    
    X_val = np.zeros(shape=(len(df_val), INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2])).astype('uint8') 
    y_val = np.zeros(shape=(len(df_val)))
    
    i = 0
    for _, row in df_val.iterrows():  
        X_val[i] = load_image(os.path.join('data', row['center'].strip()))         
        y_val[i] = row['steering'] 
        i += 1
       
    print('data loaded in {:.0f}s'.format(time.time() - start))

    return df_train, X_val, y_val
 
#read the image from file, crop off top and bottom, and resize 
def load_image(file):
    img = cv2.imread(file)

    #chop off top and bottom 25 pixels
    #img = img[25:(img.shape[0]-25), 0:img.shape[1]]    
    #img = img[50:(img.shape[0]-25), 0:img.shape[1]] 
    #img = cv2.resize(img, (INPUT_SHAPE[1], INPUT_SHAPE[0]), interpolation=cv2.INTER_AREA)  
    
    return img

#make sure steering is between -1 and 1
def valid_steering(steering):
    if abs(steering) > 1.0:
        if steering < 0:
            steering = -1.0 
        else:
            steering = 1.0
            
    return steering
    
#randomly select center/left/right image    
def get_image_steering(row):
    r = random.randint(3)
    if r == 0:       
        img = load_image(os.path.join('data', row['center'].strip())) 
        steering = row['steering']
    elif r == 1:
        #add an offset to steering for left camera images
        img = load_image(os.path.join('data', row['left'].strip())) 
        steering = row['steering'] + 0.2
        steering = valid_steering(steering)
    else:  
        #subtract the same offset to steering for right camera images 
        img = load_image(os.path.join('data', row['right'].strip())) 
        steering = row['steering'] - 0.2     
        steering = valid_steering(steering)
   
    return img, steering            
            
#generate random shadow in random polygon in the image
def generate_shadow(img):
    x1, x2 = random.randint(0, INPUT_SHAPE[1], 2)   
    new_img = img.copy()
    
    if random.randint(2) == 0:
        #choose a random polygon
        if random.randint(2) == 0:
            poly = np.array([[x1, 0], [0, 0], [0, INPUT_SHAPE[0]], [x2, INPUT_SHAPE[0]]], dtype=np.int32)
        else:
            poly = np.array([[x1, 0], [INPUT_SHAPE[1], 0], [INPUT_SHAPE[1], INPUT_SHAPE[0]], [x2, INPUT_SHAPE[0]]], dtype=np.int32)
            
        mask = img.copy()
        cv2.fillPoly(mask, [poly], (0, 0, 0))
        alpha = random.uniform(0.5, 0.75)
        cv2.addWeighted(mask, alpha, img, 1 - alpha, 0, new_img)

    return new_img
    
def augment_image(img, steering):
    w, h = img.shape[0:2]   
    output_shape = (h, w)
        
    #randomly shift image horizontally
    x = int(random.random() * 100 - 100 / 2)
    y = 0
    transM = np.float32([[1, 0, x], [0, 1, y]])
    img = cv2.warpAffine(img, transM, output_shape)
    steering = steering + x * 0.0015 
    #make sure steering is in the range
    steering = valid_steering(steering)
    
    #randomly flip image if the steering is not too small
    if random.randint(2) == 0:
        #if abs(steering) >= 0.1:
        img = cv2.flip(img, 1)
        steering = -1.0 * steering
            
    #randomly change brightness
    r = 0.25 + random.uniform(0.2, 0.5)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img_hsv[:, :, 2] = img_hsv[:, :, 2] * r
    img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)  

    #random shadow
    img = generate_shadow(img)
    
    return img, steering    

def generate_train_data(df, batch_size = 32):    
    imgs = np.zeros((batch_size, INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2])).astype('uint8') 
    steerings = np.zeros(batch_size)
    while 1:
        for i in range(batch_size):
            index = random.randint(len(df))   
            img, steering = get_image_steering(df.iloc[index])
            img, steering = augment_image(img, steering)  
            imgs[i], steerings[i] = img, steering
                    
        yield imgs, steerings
        
    
#create the neural network    
def create_model():         
    def resize(img):
        import tensorflow as tf
        img = tf.image.resize_images(img, (66, 200))
        return img
    
    model = Sequential()
    #normalize image to [-0.5, 0.5]
    model.add(Lambda(lambda x: x / 255 - 0.5, input_shape=INPUT_SHAPE, name='Normalize'))   
    #crop off top and bottom pixels
    model.add(Cropping2D(cropping=((55, 25), (0, 0)), name='Cropping'))     
    #resize image to feed to convolutional layer
    model.add(Lambda(resize, name='Resize')) 
    #model.add(BatchNormalization(input_shape=INPUT_SHAPE, name='Normalize'))   
    model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation='relu', border_mode='valid', name='Conv1'))   
    model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation='relu', border_mode='valid', name='Conv2'))
    model.add(Dropout(0.2, name='Dropout1'))   
    model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation='relu', border_mode='valid', name='Conv3'))   
    model.add(Dropout(0.2, name='Dropout2'))    
    model.add(Convolution2D(64, 3, 3, subsample=(1,1), activation='relu', border_mode='valid', name='Conv4')) 
    model.add(Convolution2D(64, 3, 3, subsample=(1,1), activation='relu', border_mode='valid', name='Conv5'))
    model.add(Dropout(0.2, name='Dropout3'))  

    model.add(Flatten())
    model.add(Dense(100, activation='relu', name='FC1'))
    model.add(Dropout(0.2, name='Dropout4')) 
    model.add(Dense(50, activation='relu', name='FC2'))
    model.add(Dropout(0.2, name='Dropout5'))
    model.add(Dense(10, activation='relu', name='FC3'))

    model.add(Dense(1, name='Output'))

    model.compile(optimizer=Adam(), loss='mse', metrics=['mean_squared_error'])
    print(model.summary())
    
    with open('model.json', 'w') as model_arch:
        model_arch.write(model.to_json())
    print('model architecture saved to model.json')
    
    plot(model, to_file='model.png')
    
    return model
    
def train(model, data_file_path):
    start = time.time()
    df_train, X_val, y_val = load_data(data_file_path)    
    
    checkpoint = ModelCheckpoint('model-{epoch:02d}-{loss:.3f}-{val_loss:.3f}.h5', verbose=1, save_weights_only=False)

    model.fit_generator(generate_train_data(df_train, batch_size=BATCH_SIZE),
                        samples_per_epoch=SAMPLES_PER_EPOCH, nb_epoch=EPOCHS, 
                        validation_data=(X_val, y_val), callbacks=[checkpoint], 
                        verbose=1, initial_epoch=0, nb_worker=8, pickle_safe=True)
    print('model training completed in {:.0f}s'.format(time.time() -start))
    
if __name__ == '__main__':
    data_file_path = os.path.join('data', 'driving_log.csv')        
    model = create_model()
    train(model, data_file_path)
    
    gc.collect()
    backend.clear_session()




