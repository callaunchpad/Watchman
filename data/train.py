import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model

import pandas as pd
from glob import glob
import os
from PIL import Image
from sklearn.model_selection import train_test_split
import numpy as np

WIDTH, HEIGHT, CHANNELS = (1024, 768, 3)
UNROLLED = 34

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(HEIGHT, WIDTH, CHANNELS))
    x = base_model.output
    x = Flatten()(x)
    x = Dense(UNROLLED, activation='sigmoid')(x)
    model = Model(inputs = base_model.input, outputs = x)
    model.compile(optimizer=Adam(lr=0.01), loss='mse')

# note: right now, the order of the output labels matters. Later we can make a network where this isn't the case

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, df, y, batch_size=32, shuffle=True):
        """
        Directory contains an "images" folder and a "outputs" folder.
        each image has name formatted like "00000001.jpg"
        each label has name formatted like "output00000001.csv"
        """
        self.batch_size = batch_size
        self.df = df
        self.y = y
        self.indices = range(len(df))
        assert len(self.df) == self.y.shape[0]
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return len(self.df) // self.batch_size

    def __getitem__(self, index):
        indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        
        return self.get_data(indices)

    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1)
    
    def get_index(self, idx):
        row = df.iloc[idx]
        X = np.array(Image.open(row.homo)) / 255.0
        y = self.y[row._name].flatten()
        
        return X, y
    
    def get_data(self, indices):
        X = np.empty((len(indices), HEIGHT, WIDTH, 3))
        y = np.empty((len(indices), UNROLLED))
        
        for i, index in enumerate(indices):
            X[i], y[i] = self.get_index(index)

        return X, y
    
    @staticmethod
    def splits(df, y, train_size=0.5):
        train, test = train_test_split(df, train_size=train_size)
        return DataGenerator(train, y[train.index]), DataGenerator(test, y[test.index])

df = pd.read_pickle('table.pkl')
y = np.load('y.npy')
train_generator, validation_generator = DataGenerator.splits(df, y, train_size=0.9)

NUM_EPOCHS=50

model.fit(
    train_generator,
    steps_per_epoch = len(train_generator),
    validation_data = validation_generator,
    validation_steps = len(validation_generator),
    epochs=NUM_EPOCHS
)

model.save('model.h5')
