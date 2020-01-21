import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GRU, Input, Masking, Dropout, Dense, Activation, Conv1D
from tensorflow.compat.v1.keras.initializers import TruncatedNormal
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K


def f1(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2 * p * r / (p + r + K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)


def f1_loss(y_true, y_pred):
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2 * p * r / (p + r + K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return 1 - K.mean(f1)


class SimpleRNNModel:
    def __init__(self, data_loader):
        self.data_loader = data_loader

        self.build_model()

    def build_model(self):

        model_input = Input((None, self.data_loader.train_x.shape[2]))
        x = model_input

        x = Masking(mask_value=0.0)(x)
        x = Conv1D(filters=32, kernel_size=5,
                   strides=1, padding="causal",
                   activation='relu', input_shape=(None, 1))(x)

        rnn_layers = [64]
        for idx, node in enumerate(rnn_layers):
            return_sequences = False if idx == len(rnn_layers) - 1 else True

            x = GRU(node,
                    return_sequences=return_sequences)(x)
            x = Dropout(0.3)(x)

        x = Dense(1, kernel_initializer=TruncatedNormal(stddev=0.01))(x)

        model_output = Activation('relu')(x)
        loss = f1_loss
        # loss = 'binary_crossentropy'

        optimizer = Adam(learning_rate=1e-8)

        model = Model(model_input, model_output)
        model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy', f1])

        self.model = model

    def load(self, path):
        model_path = tf.train.latest_checkpoint(path)
        if model_path is None:
            file_name = sorted([file_name for file_name in os.listdir(path) if file_name.endswith('.hdf5')])[-1]
            model_path = os.path.join(path, file_name)

        self.model = load_model(model_path)

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


class HurcyModel:
    def __init__(self, data_loader, class_weight=None, is_bias=False):
        self.data_loader = data_loader
        total = data_loader.get_train_data()[1].shape[0]
        pos = data_loader.get_train_data()[1].sum()
        neg = total - pos
        if is_bias:
            initial_bias = np.log([pos / neg])
        else:
            initial_bias = None
        if isinstance(class_weight, str) and class_weight == 'balanced':
            weight_for_0 = (1 / neg) * (total) / 2.0
            weight_for_1 = (1 / pos) * (total) / 2.0
            self.class_weight = {0: weight_for_0, 1: weight_for_1}
        elif isinstance(class_weight, dict):
            self.class_weight = class_weight
        else:
            self.class_weight = None
        self.build_model(initial_bias)

    def build_model(self, output_bias=None):
        if output_bias is not None:
            output_bias = tf.keras.initializers.Constant(output_bias)
        model_input = Input((None, self.data_loader.train_x.shape[2]))
        x = model_input

        x = Masking(mask_value=0.0)(x)

        rnn_layers = [64]
        for idx, node in enumerate(rnn_layers):
            return_sequences = False if idx == len(rnn_layers) - 1 else True

            x = GRU(node,
                    dropout=0.01,
                    recurrent_dropout=0.01,
                    return_sequences=return_sequences)(x)
            x = Dropout(0.4)(x)

        x = Dense(1, kernel_initializer=TruncatedNormal(stddev=0.01),
                  bias_initializer=output_bias)(x)

        model_output = Activation('sigmoid')(x)
        loss = f1_loss
        # loss = 'binary_crossentropy'

        optimizer = Adam(learning_rate=0.02)

        model = Model(model_input, model_output)
        model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy', f1])

        self.model = model

    def load(self, path):
        model_path = tf.train.latest_checkpoint(path)
        if model_path is None:
            file_name = sorted([file_name for file_name in os.listdir(path) if file_name.endswith('.hdf5')])[-1]
            model_path = os.path.join(path, file_name)

        self.model = load_model(model_path)

    def train(self, train_data, valid_data, epochs=10, verbose=0, batch_size=32, callbacks=[]):
        return self.model.fit(train_data[0], train_data[1],
                              epochs=epochs,
                              verbose=verbose,
                              batch_size=batch_size,
                              validation_data=valid_data,
                              class_weight=self.class_weight,
                              callbacks=callbacks
                              )

    def predict(self, infer_x, batch_size=None):
        return self.model.predict(infer_x, batch_size=batch_size)
