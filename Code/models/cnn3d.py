# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 19:19:19 2018

@author: Pooventhiran Naveen
"""
import keras
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Flatten
from keras.layers import Conv3D, MaxPooling3D, BatchNormalization
from sklearn.metrics import classification_report
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

class CNN:
    def __init__(self):
        print('---3D CNN for Visual Speech Recognition---')
        self.model = Sequential()
        self.history = None
        
    def build_net(self, load_name = None):
        if load_name:
            self.model = load_model('{}.h5'.format(load_name))
        else:
            self.model.add(Conv3D(name = 'conv1',
                                  filters = 64,
                                  kernel_size = 3,
                                  strides = 1,
                                  input_shape = (15, 30, 48, 1),
                                  data_format = 'channels_last'))
            self.model.add(BatchNormalization())
            self.model.add(Activation('relu'))
            self.model.add(MaxPooling3D(name = 'pool1',
                                        pool_size = (1, 3, 3),
                                        strides = (1, 2, 2)))
            self.model.add(Conv3D(name = 'conv2',
                                  filters = 128,
                                  kernel_size = 3,
                                  strides = 1,
                                  data_format = 'channels_last'))
            self.model.add(BatchNormalization())
            self.model.add(Activation('relu'))
            self.model.add(MaxPooling3D(name = 'pool2',
                                        pool_size = (1, 3, 3),
                                        strides = (1, 2, 2)))
            self.model.add(Conv3D(name = 'conv3',
                                  filters = 256,
                                  kernel_size = 3,
                                  strides = 1,
                                  data_format = 'channels_last'))
            self.model.add(BatchNormalization())
            self.model.add(Activation('relu'))
            self.model.add(MaxPooling3D(name = 'pool3',
                                        pool_size = (1, 3, 3),
                                        strides = (1, 2, 2)))
            """self.model.add(Conv3D(name = 'conv4',
                                  filters = 128,
                                  kernel_size = 2,
                                  strides = 1,
                                  activation = 'relu'))"""
            self.model.add(Flatten(name = 'flatten'))
            #self.model.add(Dense(256, name='fc5'))
            #self.model.add(Dense(128, name = 'fc6'))
            self.model.add(Dense(128, name = 'fc7'))
            self.model.add(BatchNormalization())
            #self.model.add(Dense(32, name = 'fc8'))
            self.model.add(Dense(10, name = 'fc9'))
            self.model.add(Activation('softmax', name = 'softmax'))
            
        self.model.compile(loss = keras.losses.categorical_crossentropy,
                           optimizer = keras.optimizers.SGD(),
                           metrics = ['categorical_accuracy'])
        
    def train(self, X, Y, epochs, validation_split, save = False, name = None):
        print('---Training CNN3D---')
        self.history = self.model.fit(x = X,
                                      y = Y,
                                      epochs = epochs,
                                      validation_split = validation_split).history
        if save:
            self.model.save('{}.h5'.format(name))
        
    def plot_graph(self):
        acc = self.history['categorical_accuracy']
        val_acc = self.history['val_categorical_accuracy']
        loss = self.history['loss']
        val_loss = self.history['val_loss']
        epochs = list(range(len(acc)))
        plt.plot(epochs, acc, 'b', label = 'Training accuracy')
        plt.plot(epochs, val_acc, 'g', label = 'Validation accuracy')
        plt.title('Training and Validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend(loc = 'upper left')
        plt.figure()
        plt.plot(epochs, loss, 'b', label = 'Training loss')
        plt.plot(epochs, val_loss, 'g', label = 'Validation loss')
        plt.title('Training and Validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend(loc = 'upper left')
        plt.show()
    
    def test(self, X, steps = None):
        print('---Testing CNN3D---')
        predictions = self.model.predict(x = X,
                           steps = steps)
        return predictions
    
    def evaluate(self, X, Y):
        print('---Evaluating CNN3D---')
        eval_ = self.model.evaluate(x = X, y = Y)
        print('Test loss:', eval_[0])
        print('Test accuracy:', eval_[1])
        
    def save_report(self, name, test_y, predictions, num_classes):
        target_names = ['Class {}:'.format(i) for i in range(1, num_classes)]
        with open(name, 'w') as report:
            report_ = classification_report(test_y, predictions, target_names = target_names)
            report.write(report_)
        print('Report saved successfully')
        
    def summary(self):
        return self.model.summary()