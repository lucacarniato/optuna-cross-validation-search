import tensorflow as tf
import abc
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np

class ModelKerasBase:

    def __init__(self,train_shape, num_classes):
        self.name = "keras_model"
        self.train_x_shape = train_shape
        self.num_classes = num_classes
        tf.random.set_seed(42)

    def fit(self, X, y, class_weight, X_val=None, y_val=None):

        callbacks = [EarlyStopping(monitor="val_accuracy", patience=50, mode="max", restore_best_weights=True)]
        self.model.fit(X, y,
                       batch_size=256,
                       epochs=1000,
                       callbacks=callbacks,
                       validation_data=(X_val, y_val),
                       class_weight=class_weight,
                       verbose=0)

    def predict(self, X):
        return np.argmax(self.model.predict(X), axis=-1)


    @abc.abstractmethod
    def set_params(self, parameters):
        return None


