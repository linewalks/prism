import os
import sys
import numpy as np
import pandas as pd

import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from data_loader import DataLoader
from model import SimpleRNNModel
from tensorflow.keras.callbacks import TensorBoard
from keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.metrics import f1_score, roc_auc_score
import datetime


try:
    data_path = sys.argv[1]
except IndexError:
    data_path = './data'

task_id = os.environ.get('ID')
if task_id is None:
  task_id = 'local_test'

task_path = os.path.join(data_path, 'volume', task_id)
log_path = os.path.join(data_path, 'volume', 'logs')
task_log_path = os.path.join(log_path, task_id)

if not os.path.exists(task_path):
  os.mkdir(task_path)
if not os.path.exists(log_path):
  os.mkdir(log_path)
if not os.path.exists(task_log_path):
  os.mkdir(task_log_path)

print("Train Start")

data_loader = DataLoader(data_path=os.path.join(data_path, 'train'),
                         common_path=os.path.join(data_path, 'volume'),
                         is_train = True,
                         task_path=task_path)

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, data_loader, fraction, repeat):
        'Initialization'
        self.xt, self.yt,self.nx, self.ny = data_loader()
        self.fraction = fraction
        self.repeat = repeat
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.nx) / 5))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
            #positive_valid patient중 negative data undersampling    
        rand_false2 = np.random.choice(self.nx.shape[0], size=int(np.floor(self.nx.shape[0]*self.fraction)))
        random_nx = self.nx[rand_false2]
        random_ny = self.ny[rand_false2]
        
        xt = np.repeat(self.xt, self.repeat, axis=0)
        yt = np.repeat(self.yt, self.repeat, axis=0)
        
        train_x = np.concatenate([xt,random_nx], axis=0)
        train_y = np.concatenate([yt,random_ny], axis=0)
            
        if len(train_x) == len(train_y):
            p = np.random.permutation(len(train_x))
            train_x = train_x[p]
            self.train_y = train_y[p]  
        else:
            print("there is non match")
        self.train_x = pad_sequences(train_x, padding='post', value=-5)
        return (self.train_x, self.train_y)   
    
    def shape(self):
        return self.train_x.shape
    
callbacks = [
    ModelCheckpoint(filepath=os.path.join(task_path, 'model-{epoch:02d}-{val_loss:.2f}.hdf5'),
                    monitor='val_loss',
                    mode='min',
                    save_best_only=True,
                    save_weights_only=False,
                    verbose=True
    ),
    TensorBoard(log_dir=task_log_path,
                write_graph=True
    ),
     EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=2, mode='auto')

]
 # data generation 
traingen = DataGenerator(data_loader.get_train_data,fraction = 0.1, repeat = 5)
valid_gen = DataGenerator(data_loader.get_valid_data,fraction = 0.1, repeat = 5)

sample_x,sample_y = traingen.__getitem__(1)
print("sample_x shape", sample_x.shape)
print("sample_y positive percents", sample_y.sum()/len(sample_y))

print("time before model train",datetime.datetime.now())
model = SimpleRNNModel(shape=sample_x.shape[2])
del sample_x #memory save 

# model train 

model.train(traingen, valid_gen, epochs=50, valid_steps = 10, 
            step_epoch = 10, verbose=2, callbacks=callbacks, workers=-1)

# Valid F1 score가 가장 잘나오는 베스트 

valid_x, valid_y = data_loader.prediction_data()
print("valid_x shape", valid_x.shape)
print("valid_x shape", valid_y.sum()/len(valid_y))

y_pred = model.predict(valid_x)

f1_list = []
thr_list = []
for thr in np.linspace(0, 1, 100):
  f1_list.append(f1_score(valid_y, y_pred > thr))
  thr_list.append(thr)

thr_idx = np.argmax(f1_list)
score_f1 = np.max(f1_list)
score_auroc = roc_auc_score(valid_y, y_pred)
print("Best Valid F1 :", score_f1, thr_list[thr_idx])
print("Valid AUROC :", score_auroc)

if score_auroc + score_f1 == 0:
  score = 0
else:
  score = 2 * score_auroc * score_f1 / (score_auroc + score_f1)
print("total Score : ", score)

np.save(os.path.join(task_path, 'f1.npy'), f1_list)
np.save(os.path.join(task_path, 'thr.npy'), thr_list)

print("Train Done")
