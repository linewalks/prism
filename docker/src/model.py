import os
import tensorflow as tf
from keras.layers import GRU
from keras.optimizers import adam
from keras.callbacks import History
from keras.layers import Input, Dense, Masking, Dropout
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.layers.normalization import BatchNormalization
from keras.initializers import TruncatedNormal
import numpy as np

class SimpleRNNModel:
  def __init__(self, shape):
    self.shape = shape
    self.build_model()

  def build_model(self):
    
    model_input = Input((None, self.shape))
    x = model_input

    x = Masking(mask_value=-5)(x)
    rnn_layers = [32]
    for idx, node in enumerate(rnn_layers):
      return_sequences = False if idx == len(rnn_layers) - 1 else True

      x = GRU(node,activation = 'relu',
              return_sequences=return_sequences)(x)
      x = BatchNormalization()(x)
      x = Dropout(0.5)(x)

    model_output = Dense(1, activation = 'sigmoid', kernel_initializer=TruncatedNormal(stddev=0.01))(x)

    loss = 'binary_crossentropy'

    model = Model(model_input, model_output)
    model.compile(optimizer=adam(lr =0.003), loss=loss, metrics=['accuracy'])
    print(model.summary())
    self.model = model

  def load(self, path):
    model_path = tf.train.latest_checkpoint(path)
    if model_path is None:
      file_name = sorted([file_name for file_name in os.listdir(path) if file_name.endswith('.hdf5') and file_name.startswith('model') ])[-1]
      model_path = os.path.join(path, file_name)
    
    self.model.load_weights(model_path)

  def train(self, traingen, valid_gen, epochs, valid_steps, step_epoch, verbose, callbacks,workers):

    self.model.fit_generator(generator= traingen, validation_data=valid_gen,
            steps_per_epoch=step_epoch, validation_steps = valid_steps, epochs = epochs,
                    verbose=verbose, callbacks=callbacks,workers=workers)
                                           
  def predict(self, infer_x):
    return self.model.predict(infer_x)


class Autoencoder:
  def __init__(self, train_measure):
    self.train_measure = train_measure
    self.build_model()

  def build_model(self):
    
    #인코딩될 표현(representation)의 크기
#     encoding_dims=[3,6,9,12]
    # 입력 플레이스홀더
    input_img = Input(shape=(self.train_measure.shape[1],))
    # "encoded"는 입력의 인코딩된 표현
    encoded1 = Dense(128, activation='tanh', name='encoder1' )(input_img)
    encoded2 = Dense(128, activation='relu', name='encoder2' )(encoded1)

    # "decoded"는 입력의 손실있는 재구성 (lossy reconstruction)
    decoded1 = Dense(self.train_measure.shape[1], activation='tanh', name='decoder1')(encoded2)

    # 입력을 입력의 재구성으로 매핑할 모델
    autoencoder = Model(input_img, decoded1)

    autoencoder.compile(loss='mean_squared_error',optimizer=adam(lr = 0.003))

    print(autoencoder.summary())

    self.model = autoencoder

  def load(self, path):
    model_path = tf.train.latest_checkpoint(path)
    if model_path is None:
      file_name = sorted([file_name for file_name in os.listdir(path) if file_name.endswith('.hdf5') and file_name.startswith('encoder')])[-1]
      model_path = os.path.join(path, file_name)
    
    self.model.load_weights(model_path)

  def train(self, train_measure, valid_measure, epochs, batch_size, verbose, callbacks):

    self.model.fit(train_measure, train_measure, batch_size=batch_size, 
                                        epochs=epochs, verbose=verbose, 
                                        validation_data=(valid_measure, valid_measure),
                                        callbacks=callbacks)
                              
  def predict(self, total_measure):
   
    input_img = Input(shape=(self.train_measure.shape[1],))
    layer1=self.model.layers[1]
    layer2=self.model.layers[2]

    encoder= Model(input_img, layer2(layer1(input_img)))
    output=encoder.predict(total_measure)
    return output
  
  