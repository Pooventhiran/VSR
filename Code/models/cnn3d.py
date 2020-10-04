# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 19:19:19 2018

@author: Pooventhiran Naveen
"""
import tensorflow.keras
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout
from tensorflow.keras.layers import Conv3D, MaxPooling3D, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report


class CNN:
    def __init__(self):
        print("---3D CNN for Visual Speech Recognition---")
        self.model = Sequential()
        self.history = None

    def build_net(self):
        self.model.add(
            Conv3D(
                name="conv1",
                filters=8,
                kernel_size=3,
                strides=1,
                input_shape=(15, 30, 48, 1),
                data_format="channels_last",
            )
        )
        self.model.add(BatchNormalization())
        self.model.add(Activation("relu"))
        self.model.add(
            Conv3D(
                name="conv2",
                filters=16,
                kernel_size=3,
                strides=1,
                data_format="channels_last",
            )
        )
        self.model.add(BatchNormalization())
        self.model.add(Activation("relu"))
        self.model.add(
            Conv3D(
                name="conv3",
                filters=32,
                kernel_size=3,
                strides=1,
                data_format="channels_last",
            )
        )
        self.model.add(BatchNormalization())
        self.model.add(Activation("relu"))
        self.model.add(
            MaxPooling3D(name="pool1", pool_size=(1, 3, 3), strides=(1, 2, 2))
        )
        self.model.add(Flatten(name="flatten"))
        self.model.add(
            Dense(
                32,
                name="fc1",
                kernel_regularizer=tensorflow.keras.regularizers.l2(1e-2),
            )
        )
        self.model.add(BatchNormalization())
        self.model.add(Activation("relu"))
        self.model.add(Dropout(0.2))
        self.model.add(
            Dense(
                10,
                name="fc2",
                kernel_regularizer=tensorflow.keras.regularizers.l2(1e-2),
            )
        )
        self.model.add(Activation("softmax", name="softmax"))

        self.model.compile(
            loss=tensorflow.keras.losses.categorical_crossentropy,
            optimizer=tensorflow.keras.optimizers.SGD(),
            metrics=[
                "categorical_accuracy",
                tensorflow.keras.metrics.Precision(),
                tensorflow.keras.metrics.Recall(),
            ],
        )

    def train(
        self, X, Y, epochs, validation_split, batch_size=32, save=False, name=None
    ):
        self.weight_file = name
        print("---Training CNN3D---")
        es = EarlyStopping("val_loss", mode="min", patience=100, verbose=1)
        mc = ModelCheckpoint(
            f"../{self.weight_file}",
            monitor="val_categorical_accuracy",
            mode="max",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        )
        self.history = self.model.fit(
            x=X,
            y=Y,
            shuffle=True,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split,
            callbacks=[es, mc] if save else None,
        ).history

    def plot_graph(self):
        acc = self.history["categorical_accuracy"]
        val_acc = self.history["val_categorical_accuracy"]
        loss = self.history["loss"]
        val_loss = self.history["val_loss"]
        epochs = list(range(len(acc)))
        plt.plot(epochs, acc, "b", label="Training accuracy")
        plt.plot(epochs, val_acc, "g", label="Validation accuracy")
        plt.title("Training and Validation accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend(loc="upper left")
        plt.figure()
        plt.plot(epochs, loss, "b", label="Training loss")
        plt.plot(epochs, val_loss, "g", label="Validation loss")
        plt.title("Training and Validation loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend(loc="upper left")
        plt.show()

    def test(self, X):
        print("---Testing CNN3D---")
        if self.weight_file is not None:
            self.model.load_weights("../{self.weight_file")
        predictions = self.model.predict(x=X)
        return predictions

    def evaluate(self, X, Y):
        print("---Evaluating CNN3D---")
        if self.weight_file is not None:
            self.model.load_weights("../{self.weight_file}")
        eval_ = self.model.evaluate(x=X, y=Y)
        print("Test loss:", eval_[0])
        print("Test accuracy:", eval_[1])
        print("All metrics:", eval_[2:])

    def print_report(self, truth, predictions, labels):
        import seaborn as sn
        import pandas as pd

        targets = sorted(labels.items(), key=lambda kv: k)
        ax = sn.heatmap(
            cf,
            annot=True,
            fmt="d",
            xticklabels=[kv[1] for kv in targets],
            yticklabels=[kv[1] for kv in targets],
        )
        ax.set(xlabel="Prediction", ylabel="Ground Truth")
        plt.show()
        print(
            classification_report(
                truth, predictions, digits=2, target_names=[kv[1] for kv in targets]
            )
        )

    def summary(self):
        return self.model.summary()
