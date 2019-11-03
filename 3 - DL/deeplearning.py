from tensorflow.keras.callbacks import Callback
from keras.datasets import cifar100
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from classification_models.resnet import ResNet18
import numpy as np
import plotly.graph_objs as go
import plotly.offline as offline
import warnings

warnings.simplefilter("ignore")

#ATTENTION! Resizing images for the 224x224 of the ResNet is done by uploading the images from a folder
# in .flow_from_directory. The resizing process of such a large dataset may cause a memory errore while fitting

# Select Class: NN (basic neural network), CNN (convolutional neural netowrk),
# CNN_mod (convolutional neural network with batch normalization and Dropout)
# or ResNet (pretrained ResNet18 finetuned)
SELECT = "CNN_mod"

# Select Augmentation for CNN: crop (random crop) or flip (random horizontal flipping)
AUG = ""

if (SELECT == "ResNet"):
    AUG = "flip"

# constant
IMG_CHANNELS = 3
IMG_ROWS = 32
IMG_COLS = 32

BATCH_SIZE = 256
NB_EPOCH = 20
NB_CLASSES = 100

if (SELECT == "ResNet"):
    BATCH_SIZE = 128
    NB_EPOCH = 10

# Plot results callback
class Plot(Callback):
    def __init__(self, test_data):
        self.test_data = test_data

    def on_train_begin(self, logs={}):
        self.i = 1
        self.x = []
        self.losses = []
        self.acc = []
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        x, y = self.test_data
        loss, accuracy = self.model.evaluate(x, y, verbose=0)
        print("Testing loss:", loss, "accuracy:", accuracy)

        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.acc.append(accuracy)
        self.i += 1

    def on_train_end(self, epoch, logs={}):
        trace_loss = go.Scatter(x=self.x, y=self.losses, name="loss")
        trace_acc = go.Scatter(x=self.x, y=self.acc,
                               name="accuracy", yaxis='y2')
        data = [trace_loss, trace_acc]

        layout = dict(title='Loss and accuracy in each epoch',
                      xaxis=dict(title='Epoch'),
                      yaxis=dict(title='loss (on training set)'),
                      yaxis2=dict(title='accuracy (on test set)', overlaying='y', side='right'))
        fig = dict(data=data, layout=layout)
        offline.plot(fig)

# normalization function
def normalize(X_train, X_test):
    X_train /= 255
    X_test /= 255
    mean = np.mean(X_train, axis=(0, 1, 2, 3))
    std = np.std(X_train, axis=(0, 1, 2, 3))
    X_train = (X_train - mean) / (std + 1e-7)
    X_test = (X_test - mean) / (std + 1e-7)
    return X_train, X_test

# crop function
def random_crop(img):
    img = transform.resize(img, (40, 40, 3))
    assert img.shape[2] == 3
    height, width = img.shape[0], img.shape[1]
    dy = 32
    dx = 32
    x = np.random.randint(0, width - dx + 1)
    y = np.random.randint(0, height - dy + 1)
    return img[y:(y + dy), x:(x + dx), :]



# RUNNING CODE FROM HERE:
# load dataset
(X_train, y_train), (X_test, y_test) = cifar100.load_data()
print("X_train shape:", X_train.shape)
print(X_train.shape[0], "train samples")
print(X_test.shape[0], "test samples")

# convert to categorical
Y_train = np_utils.to_categorical(y_train, NB_CLASSES)
Y_test = np_utils.to_categorical(y_test, NB_CLASSES)

# image preprocessing
X_train = X_train.reshape(
    X_train.shape[0], IMG_ROWS, IMG_COLS, IMG_CHANNELS).astype('float32')
X_test = X_test.reshape(
    X_test.shape[0], IMG_ROWS, IMG_COLS, IMG_CHANNELS).astype('float32')
X_train, X_test = normalize(X_train, X_test)

#X_train = resize_img(X_train)
if (AUG == "crop"):
    datagen = ImageDataGenerator(preprocessing_function = random_crop)
    train_aug = datagen.flow(X_train, Y_train, batch_size=BATCH_SIZE)
if (SELECT != "ResNet") and (AUG == "flip"):
    datagen = ImageDataGenerator(horizontal_flip=True)
    train_aug = datagen.flow(X_train, Y_train, batch_size=BATCH_SIZE)
if (SELECT == "ResNet") and (AUG == "flip"):
    datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True)
    train_aug = datagen.flow_from_directory("D:/cifar100/train", target_size=(224, 224), batch_size=BATCH_SIZE, class_mode='categorical')
    datagen_test = ImageDataGenerator(rescale=1./255)
    test_aug = datagen.flow_from_directory("D:/cifar100/test", target_size=(224, 224), batch_size=BATCH_SIZE, class_mode='categorical')

# MODELS
if (SELECT == "NN"):
    # Traditional Neural Network
    model = Sequential()
    model.add(Flatten(input_shape=(IMG_ROWS, IMG_COLS, IMG_CHANNELS)))
    model.add(Dense(4096))
    model.add(Activation('sigmoid'))
    model.add(Dense(4096))
    model.add(Activation('sigmoid'))
    model.add(Dense(NB_CLASSES))
    model.add(Activation('softmax'))
    model.summary()

if (SELECT == "CNN"):
    # Convolutional Neural Network
    model = Sequential()
    model.add(Conv2D(128, kernel_size=5, strides=2, padding='valid',
                     input_shape=(IMG_ROWS, IMG_COLS, IMG_CHANNELS)))
    model.add(Activation('relu'))
    model.add(Conv2D(128, kernel_size=3, strides=1, padding='valid'))
    model.add(Activation('relu'))
    model.add(Conv2D(128, kernel_size=3, strides=1, padding='valid'))
    model.add(Activation('relu'))
    model.add(Conv2D(256, kernel_size=3, strides=1, padding='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=2, strides=2, padding='valid'))
    model.add(Flatten())
    model.add(Dense(4096))
    model.add(Activation('relu'))
    model.add(Dense(NB_CLASSES))
    model.add(Activation('softmax'))
    model.summary()

if (SELECT == "CNN_mod"):
    # Convolutional Neural Network with BatchNormalization and Dropout
    model = Sequential()
    model.add(Conv2D(128, kernel_size=5, strides=2, padding='valid',
                     input_shape=(IMG_ROWS, IMG_COLS, IMG_CHANNELS)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(128, kernel_size=3, strides=1, padding='valid'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(128, kernel_size=3, strides=1, padding='valid'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(256, kernel_size=3, strides=1, padding='valid'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=2, strides=2, padding='valid'))
    model.add(Flatten())
    model.add(Dense(4096))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NB_CLASSES))
    model.add(Activation('softmax'))
    model.summary()

if (SELECT == "ResNet"):
    # ResNet18
    model = Sequential()
    model.add(ResNet18(input_shape=(224, 224, 3), weights='imagenet', include_top=False))
    model.add(Flatten())
    model.add(Dense(NB_CLASSES))
    model.add(Activation('softmax'))
    model.summary()

# train
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.0001), metrics=['accuracy'])
if (AUG == "flip") or (AUG == "crop") or (SELECT == "ResNet"):
    model.fit_generator(train_aug, steps_per_epoch=X_train.shape[0] // BATCH_SIZE, epochs=NB_EPOCH, verbose=1,
              shuffle=True, callbacks=[Plot((X_test, Y_test))])
else:
    model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=NB_EPOCH, verbose=1,
              shuffle=True, callbacks=[Plot((X_test, Y_test))])

score = model.evaluate(X_test, Y_test, batch_size=BATCH_SIZE, verbose=1)
print("Final test score:", score[0])
print("Final test accuracy:", score[1])
