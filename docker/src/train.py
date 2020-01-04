import os
import sys
import numpy as np
import pandas as pd

from data_loader import DataLoader
from model import Autoencoder
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import f1_score, roc_auc_score

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
                         task_path=task_path)

print("train_x shape : ", data_loader.train_x.shape)

autoencoder = Autoencoder(data_loader, training = True)

callbacks = [
    ModelCheckpoint(filepath=os.path.join(task_path, 'encoder-{epoch:02d}-{val_loss:2f}.hdf5'),
                    monitor='val_loss',
                    mode='min',
                    save_best_only=True,
                    save_weights_only=False,
                    verbose=True
    )
]

autoencoder.train(data_loader.get_train_data(), data_loader.get_valid_data(),
            verbose=0,
            epochs=20, batch_size=32,
            callbacks=callbacks)

autoencoder.RNNmodel()

callbacks = [
    ModelCheckpoint(filepath=os.path.join(task_path, 'model-{epoch:02d}-{val_loss:2f}.hdf5'),
                    monitor='val_loss',
                    mode='min',
                    save_best_only=True,
                    save_weights_only=False,
                    verbose=True
    )
]
autoencoder.rnntrain(data_loader.get_train_data(), data_loader.get_valid_data(),
            verbose=0,
            epochs=20, batch_size=32,
            callbacks=callbacks)


# Valid F1 score가 가장 잘나오는 베스트 

valid_x, valid_y = data_loader.get_valid_data()
y_pred = autoencoder.predict(valid_x)

f1_list = []
thr_list = []
for thr in np.linspace(0, 1, 100):
  f1_list.append(f1_score(valid_y, y_pred > thr))
  thr_list.append(thr)

thr_idx = np.argmax(f1_list)
score_f1 = np.max(f1_list)
score_auroc = roc_auc_score(valid_y, y_pred)
print("Best Valid F1", score_f1, thr_list[thr_idx])
print("Valid AUROC", score_auroc)

if score_auroc + score_f1 == 0:
  score = 0
else:
  score = 2 * score_auroc * score_f1 / (score_auroc + score_f1)
print("Valid Score", score)

np.save(os.path.join(task_path, 'f1.npy'), f1_list)
np.save(os.path.join(task_path, 'thr.npy'), thr_list)

print("Train Done")
