import os
import sys
import numpy as np
import pandas as pd

from data_loader import DataLoader
from model import SimpleRNNModel, f1, f1_loss
from tensorflow.keras.utils import get_custom_objects


def count_elements(seq) -> dict:
  """Tally elements from `seq`."""
  hist = {}
  for i in seq:
    hist[i] = hist.get(i, 0) + 1
  return hist


def ascii_histogram(seq) -> None:
  """A horizontal frequency-table/histogram plot."""
  counted = count_elements(seq)
  for k in sorted(counted):
    print('{0:5d} {1}'.format(k, '+' * counted[k]))


data_path = sys.argv[1]

task_id = os.environ.get('ID')
if task_id is None:
  task_id = 'local_test'

task_path = os.path.join(data_path, 'volume', task_id)

print("Inference Start")

data_loader = DataLoader(data_path=os.path.join(data_path, 'test'),
                         common_path=os.path.join(data_path, 'volume'),
                         task_path=task_path,
                         is_train=False)
model = SimpleRNNModel(data_loader)

# 모델 로드

get_custom_objects().update({"f1_loss": f1_loss})
get_custom_objects().update({"f1": f1})

model.load(task_path)

# threhshold
f1_list = np.load(os.path.join(task_path, 'f1.npy'))
thr_list = np.load(os.path.join(task_path, 'thr.npy'))
# thr = thr_list[np.argmax(f1_list)]
thr = 0.02
y_key = data_loader.key
y_pred = model.predict(data_loader.get_infer_data())
y_pred = pd.DataFrame(y_pred, columns=['pred'])

y_pred = pd.concat([y_key, y_pred], axis=1)

out_key = data_loader.cohort_df[['SUBJECT_ID', 'COHORT_END_DATE']]
merged_df = pd.merge(out_key, y_pred, on=['SUBJECT_ID', 'COHORT_END_DATE'], how='left')

merged_df['LABEL_PROBABILITY'] = merged_df.pred
merged_df['LABEL'] = (merged_df['LABEL_PROBABILITY'] > thr).astype(np.int32)

merged_df[['LABEL', 'LABEL_PROBABILITY']].to_csv(os.path.join(data_path, 'output', 'output.csv'), index=False)

print("True Pred", merged_df['LABEL'].sum(), merged_df['LABEL'].sum() / merged_df.shape[0])
print("LABEL_PROBABILITY statistics")
print(merged_df[['LABEL', 'LABEL_PROBABILITY']].describe())
try:
  ranges = merged_df.LABEL_PROBABILITY.quantile([.05, .25, .5, .75, .95]).values
  print('LABEL count per probability ranges')
  print(merged_df.groupby(pd.cut(merged_df.LABEL_PROBABILITY, ranges)).count()[['LABEL']])
except ValueError:
  pass
print("Inference Done")
