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
from tensorflow.keras.models import load_model

class SimpleRNNModel:
  def __init__(self, shape):
    self.shape = shape
    self.build_model()

  def build_model(self):
    
    model_input = Input((None, self.shape))
    x = model_input

    x = Masking(mask_value=-5)(x)
    x = Dense(64, activation = 'tanh',name = '1stDense')(x)
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
    model.compile(optimizer=adam(lr =0.001), loss=loss, metrics=['accuracy'])
    print(model.summary())
    self.model = model

  def load(self, path):
    model_path = tf.train.latest_checkpoint(path)
    if model_path is None:
      file_name = sorted([file_name for file_name in os.listdir(path) if file_name.endswith('.hdf5')])[-1]
      model_path = os.path.join(path, file_name)
    
    self.model = load_model(model_path)

  def train(self, traingen, valid_gen, epochs, valid_steps, step_epoch, verbose, callbacks,workers):

    self.model.fit_generator(generator= traingen, validation_data=valid_gen,
            steps_per_epoch=step_epoch, validation_steps = valid_steps, epochs = epochs,
                    verbose=verbose, callbacks=callbacks,workers=workers)
                                           
  def predict(self, infer_x):
    return self.model.predict(infer_x)



  
  