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


class Autoencoder:
  def __init__(self, data_loader, training = True):
    self.data_loader = data_loader
    if training:
        self.build_model()
    else:
        self.build_model()
        self.RNNmodel()
        
  def build_model(self):
    
    #인코딩될 표현(representation)의 크기
#     encoding_dims=[3,6,9,12]
    # 입력 플레이스홀더
    input_img = Input(shape=(self.data_loader.train_x.shape[1],self.data_loader.train_x.shape[2],))
    x = GRU(32,activation = 'tanh',
              return_sequences=True)(input_img)
    
    output = Dense(self.data_loader.train_x.shape[2], activation ='tanh')(x)
    
    # 입력을 입력의 재구성으로 매핑할 모델
    autoencoder = Model(input_img, output)


    autoencoder.compile(loss='mean_squared_error',optimizer=adam(lr = 0.003))

    print(autoencoder.summary())

    self.model = autoencoder

  def load(self, path):
    model_path = tf.train.latest_checkpoint(path)
    if model_path is None:
      file_name = sorted([file_name for file_name in os.listdir(path) if file_name.endswith('.hdf5') and file_name.startswith('encoder')])[-1]
      model_path = os.path.join(path, file_name)
    
    self.model.load_weights(model_path)

  def train(self, train_data, valid_data, epochs, batch_size, verbose, callbacks):

    self.model.fit(train_data[0], train_data[0], batch_size=batch_size, 
                                        epochs=epochs, verbose=verbose, 
                                        validation_data=(valid_data[0], valid_data[0]),
                                        callbacks=callbacks)

  def rnn_load(self, path):
    model_path = tf.train.latest_checkpoint(path)
    if model_path is None:
      file_name = sorted([file_name for file_name in os.listdir(path) if file_name.endswith('.hdf5') and file_name.startswith('model')])[-1]
      model_path = os.path.join(path, file_name)
    
    self.rnnmodel.load_weights(model_path)
                              
  def RNNmodel(self):
   
    input_img = Input(shape=(self.data_loader.train_x.shape[1], self.data_loader.train_x.shape[2],))
    layer1 = self.model.layers[1]
    layer1.trainable = False
    
    x = layer1(input_img)
    x = GRU(32, activation = 'relu') (x)
    x = Dense(16, activation = 'relu') (x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    output = Dense(1, activation = 'sigmoid')(x)
    
    loss = 'binary_crossentropy'

    model = Model(input_img, output)
    model.compile(optimizer=adam(lr = 0.003), loss=loss, metrics=['accuracy'])

    self.rnnmodel = model
    
  def rnntrain(self, train_data, valid_data, epochs, batch_size, verbose, callbacks):

    self.rnnmodel.fit(train_data[0], train_data[1], batch_size=batch_size, 
                                        epochs=epochs, verbose=verbose, 
                                        validation_data=(valid_data[0], valid_data[1]),
                                        callbacks=callbacks)

  def predict(self, infer_x):
    return self.rnnmodel.predict(infer_x)