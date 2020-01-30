import os
from keras.optimizers import RMSprop, adam
from keras.callbacks import History
from keras.layers import Input, Dense, GRU
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from keras import layers
import keras


class SimpleRNNModel:
    def __init__(self, data_loader):
        self.data_loader = data_loader

        self.build_model()

    def build_model(self):

        model_input = Input(shape=(None, self.data_loader.train_x.shape[2]))
        x = layers.Masking(mask_value=-1)(model_input)

        rnn_layers = [64]
        for idx, node in enumerate(rnn_layers):
            return_sequences = False if idx == len(rnn_layers) - 1 else True

            x = layers.GRU(node, activation='tanh',
                           return_sequences=return_sequences)(x)
            x = layers.Dropout(0.3)(x)

        model_output = Dense(1, activation='sigmoid')(x)

        loss = 'binary_crossentropy'

        optimizer = adam(lr=0.001)

        model = Model(model_input, model_output)
        model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

        self.model = model

    def load(self, path):

        file_name = sorted([file_name for file_name in os.listdir(
            path) if file_name.endswith('.hdf5')])[-1]
        model_path = os.path.join(path, file_name)

        self.model = self.model.load_weights(model_path)

    def train(self, train_data, valid_data, epochs=10, verbose=0, batch_size=32, callbacks=[]):
        return self.model.fit(train_data[0], train_data[1],
                              epochs=epochs,
                              verbose=verbose,
                              batch_size=batch_size,
                              validation_data=valid_data,
                              callbacks=callbacks
                              )

    def predict(self, infer_x, batch_size=None):
        return self.model.predict(infer_x, batch_size=batch_size)