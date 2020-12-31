from ModelKerasBase import *

class ModelKerasFullyConnected(ModelKerasBase):

    def __init__(self,train_shape, num_classes):
        ModelKerasBase.__init__(self,train_shape, num_classes)

    def set_params(self, num_units, num_hidden, learning_rate, dropout):

        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.Dense(num_units, activation='relu'))
        self.model.add(tf.keras.layers.Dropout(dropout))

        for l in range(0, num_hidden):
            self.model.add(tf.keras.layers.Dense(num_units, activation='relu'))
            self.model.add(tf.keras.layers.Dropout(dropout))

        self.model.add(tf.keras.layers.Dense(self.num_classes, activation='softmax'))

        opt = tf.keras.optimizers.Adam(lr=learning_rate)
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
