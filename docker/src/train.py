import os
import sys
import numpy as np
import pandas as pd

from data_loader import DataLoader
from model import SimpleRNNModel
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn.metrics import f1_score, roc_auc_score

data_path = sys.argv[1]

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
model = SimpleRNNModel(data_loader)

callbacks = [
    ModelCheckpoint(filepath=os.path.join(task_path, 'model-{epoch:02d}-{val_loss:2f}.hdf5'),
                    monitor='val_loss',
                    checkpoint_mode='min',
                    save_best_only=True,
                    save_weights_only=False,
                    verbose=True
    ),
    TensorBoard(log_dir=task_log_path,
                write_graph=True
    )
]

model.train(data_loader.get_train_data(), data_loader.get_valid_data(),
            verbose=0,
            epochs=15, batch_size=32,
            callbacks=callbacks)

# Valid F1 score가 가장 잘나오는 베스트 

valid_x, valid_y = data_loader.get_valid_data()
y_pred = model.predict(valid_x)

f1_list = []
thr_list = []
for thr in np.linspace(0, 1, 100):
  f1_list.append(f1_score(valid_y, y_pred > thr))
  thr_list.append(thr)

thr_idx = np.argmax(f1_list)
print("Best Valid F1", np.max(f1_list), thr_list[thr_idx])
print("Valid AUROC", roc_auc_score(valid_y, y_pred))

np.save(os.path.join(task_path, 'f1.npy'), f1_list)
np.save(os.path.join(task_path, 'thr.npy'), thr_list)

print("Train Done")