from tensorflow.keras.models import Model
from tensorflow.keras.layers import GRU, Input, Masking, Dropout, Dense, Activation
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.optimizers import Adam



class SimpleRNNModel:
  def __init__(self, data_loader):
    self.data_loader = data_loader
    
    self.build_model()

  def build_model(self):
    
    model_input = Input((None, self.data_loader.train_x.shape[2]))
    x = model_input
    
    x = Masking(mask_value=0.0)(x)

    rnn_layers = [64]
    for idx, node in enumerate(rnn_layers):
      return_sequences = False if idx == len(rnn_layers) - 1 else True

      x = GRU(node,
              return_sequences=return_sequences)(x)
      x = Dropout(0.3)(x)

    x = Dense(1, kernel_initializer=TruncatedNormal(stddev=0.01))(x)

    model_output = Activation('sigmoid')(x)
    loss = 'binary_crossentropy'

    optimizer = Adam(learning_rate=0.001)

    model = Model(model_input, model_output)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    self.model = model

  def train(self, train_data, valid_data, epochs=10, verbose=0, batch_size=32):
    self.model.fit(train_data[0], train_data[1],
                   epochs=epochs,
                   verbose=verbose,
                   batch_size=batch_size,
                   validation_data=valid_data
                   )