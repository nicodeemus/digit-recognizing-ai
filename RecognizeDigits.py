from keras.datasets import mnist
from matplotlib import pyplot
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense, Flatten
from keras.layers.convolutional import *
from tensorflow.keras.optimizers import Adam
from keras.metrics import sparse_categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import load_model
import PIL.ImageOps
import numpy as nmp


def load_data():
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()


    #print('Train images: ' + str(train_images.shape))
    #print('Train labels: ' + str(train_labels.shape))
    #print('Test images:  ' + str(test_images.shape))
    #print('Test labels:  ' + str(test_labels))

    train_images = train_images / 255.0
    #test_images = test_images / 255.0

    #pyplot.figure()
    #pyplot.imshow(train_images[0])
    #pyplot.colorbar()
    #pyplot.grid(False)
    #pyplot.show()

    return train_images, train_labels


def make_model():
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(100, activation = 'relu', kernel_initializer='he_uniform'),
        Dense(10, activation='softmax'),
    ])

    model.summary()
    return model

def train_model(model, train_images, train_labels):

    model.compile(optimizer=SGD(learning_rate=0.01, momentum=0.9), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_images, train_labels, validation_split=0.20, batch_size=30, epochs=10, verbose=2)
    model.save('final_model.h5')

    return model

def predict_from_testdata(test_data, model):
    picture_index = 0
    while (picture_index < 10):
        img = test_data[picture_index]
        prediction = model.predict(img)
        predicted_digit = nmp.argmax(prediction)
        print("I guessed: %s the real value was: %d" %(predicted_digit, picture_index))
        picture_index += 1


def load_single_testdata(file):
    img = tf.keras.utils.load_img(file, color_mode = "grayscale", target_size=(28, 28))
    img = PIL.ImageOps.invert(img)
    img = tf.keras.utils.img_to_array(img)
    img = img.reshape(1, 28, 28, 1)
    img = img.astype('float32')
    img = img / 255.0

    #pyplot.figure()
    #pyplot.imshow(img[0])
    #pyplot.colorbar()
    #pyplot.grid(False)
    #pyplot.show()

    return img

def arrange_testdata ():

    test_pictures = []

    test_pictures.append(load_single_testdata("Zero.png"))
    test_pictures.append(load_single_testdata("One.png"))
    test_pictures.append(load_single_testdata("Two.png"))
    test_pictures.append(load_single_testdata("Three.png"))
    test_pictures.append(load_single_testdata("Four.png"))
    test_pictures.append(load_single_testdata("Five.png"))
    test_pictures.append(load_single_testdata("Six.png"))
    test_pictures.append(load_single_testdata("Seven.png"))
    test_pictures.append(load_single_testdata("Eight.png"))
    test_pictures.append(load_single_testdata("Nine.png"))

    return test_pictures


train_images, train_labels = load_data()
#model = load_model('final_model.h5')
model = make_model()
model = train_model(model, train_images, train_labels)
test_data = arrange_testdata()
predict_from_testdata(test_data, model)

