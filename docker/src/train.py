import os
import sys
import numpy as np
import pandas as pd

from data_loader import DataLoader
from model import SimpleRNNModel, Autoencoder, ConcatRNN
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from sklearn.metrics import f1_score, roc_auc_score, recall_score, precision_score

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

#Autoencoding

# data_loader = DataLoader(data_path=os.path.join(data_path, 'train'),
#                          common_path=os.path.join(data_path, 'volume'),
#                          autoencoder =True,
#                          task_path=task_path)

# auto_model = Autoencoder(data_loader.train_measure)
# print("measurement_df shape: ", data_loader.train_measure.shape)

# callbacks1 = [
#     ModelCheckpoint(filepath=os.path.join(task_path, 'encoder-{epoch:02d}-{val_loss:2f}.hdf5'),
#                     monitor='val_loss',
#                     mode='min',
#                     save_best_only=True,
#                     save_weights_only=False,
#                     verbose=True
#     ),
#     EarlyStopping(monitor='val_loss', min_delta=0, patience=5,
#                                    verbose=0, mode='auto')
# ]
# auto_model.train(data_loader.train_measure, data_loader.valid_measure,
#             verbose=0,
#             epochs=500, batch_size=600,
#             callbacks=callbacks1)

# del data_loader
# del auto_model
# del callbacks1

# auto_model.predict(data_loader.measure_auto)
data_loader2 = DataLoader(data_path=os.path.join(data_path, 'train'),
                         common_path=os.path.join(data_path, 'volume'),
                         autoencoder =False,
                         task_path=task_path)

# model = SimpleRNNModel(data_loader2)
model = ConcatRNN(data_loader2)
print("train_x shape: ",data_loader2.train_x.shape)
callbacks2 = [
    ModelCheckpoint(filepath=os.path.join(task_path, 'model-{epoch:02d}-{val_loss:2f}.hdf5'),
                    monitor='val_loss',
                    mode='min',
                    save_best_only=True,
                    save_weights_only=False,
                    verbose=True
    ),
    EarlyStopping(monitor='val_loss', min_delta=0, patience=5,
                                   verbose=0, mode='auto'),
    TensorBoard(log_dir=task_log_path, write_graph=True)
]

hist = model.train(data_loader2.get_train_data(), data_loader2.get_valid_data(),
            verbose=0,
            epochs=100, batch_size=200,
            callbacks=callbacks2)

keys = hist.history.keys()
print_keys = [key.replace('val_', 'v_') for key in (['epoch'] + list(keys))]
print("\t".join(print_keys))
for idx in range(len(hist.history['loss'])):
  log_list = [str(idx+1)]
  for key in keys:
    log_list.append("%.4f" % hist.history[key][idx])

  print("\t".join(log_list))

def print_score(data_x, data_y, data_type):
  y_pred = model.predict(data_x, batch_size=1024)
  f1_list = []
  thr_list = []
  for thr in np.linspace(0, 1, 100):
    f1_list.append(f1_score(data_y, y_pred > thr))
    thr_list.append(thr)

  thr_idx = np.argmax(f1_list)
  score_f1 = np.max(f1_list)
  score_auroc = roc_auc_score(data_y, y_pred)

  thr = thr_list[thr_idx]
  score_precision = precision_score(data_y, y_pred > thr)
  score_recall = recall_score(data_y, y_pred > thr)
  print("Best", data_type, "F1", score_f1)
  print("Best", data_type, "F1 Precision", score_precision)
  print("Best", data_type, "F1 Recall", score_recall)
  print("Best", data_type, "F1 Threshold", thr)
  print(data_type, "AUROC", score_auroc)

  if score_auroc + score_f1 == 0:
    score = 0
  else:
    score = 2 * score_auroc * score_f1 / (score_auroc + score_f1)
  print(data_type, "Score", score)

  return f1_list, thr_list


train_x, train_y = data_loader2.get_train_data()
valid_x, valid_y = data_loader2.get_valid_data()

train_f1, train_thr = print_score(train_x, train_y, 'Train')
valid_f1, valid_thr = print_score(valid_x, valid_y, 'Valid')

np.save(os.path.join(task_path, 'f1.npy'), valid_f1)
np.save(os.path.join(task_path, 'thr.npy'), valid_thr)

print("Train Done")
