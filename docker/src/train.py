import os
import sys
import pandas as pd

from data_loader import DataLoader
from model import SimpleRNNModel
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

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

data_loader = DataLoader(data_path=os.path.join(data_path, 'train'))
model = SimpleRNNModel(data_loader)

callbacks = [
    ModelCheckpoint(filepath=os.path.join(task_path, 'model-{epoch:02d}-{val_loss:2f}.hdf5'),
                    monitor='val_loss',
                    checkpoint_mode='min',
                    save_best_only=True,
                    save_weights_only=False,
                    verbose=True
    ),
    TensorBoard(log_dir=os.path.join(data_path, 'volume', 'logs'),
    write_graph=True
    )
]

model.train(data_loader.get_train_data(), data_loader.get_valid_data(),
            verbose=1,
            epochs=10, batch_size=32,
            callbacks=callbacks)

