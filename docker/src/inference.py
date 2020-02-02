import os
import sys
import numpy as np
import pandas as pd

from data_loader import DataLoader
from model import SimpleRNNModel

try:
    data_path = sys.argv[1]
except IndexError:
    data_path = './data'

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
model.load(task_path)

# threhshold
f1_list = np.load(os.path.join(task_path, 'f1.npy'))
thr_list = np.load(os.path.join(task_path, 'thr.npy'))
thr = thr_list[np.argmax(f1_list)]

y_key = data_loader.key
y_pred = model.predict(data_loader.get_infer_data())
y_pred = pd.DataFrame(y_pred, columns=['pred'])

y_pred = pd.concat([y_key, y_pred], axis=1)

out_key = data_loader.cohort_df[['SUBJECT_ID', 'COHORT_END_DATE']]
merged_df = pd.merge(out_key, y_pred, on=['SUBJECT_ID', 'COHORT_END_DATE'], how='left')

merged_df['LABEL_PROBABILITY'] = merged_df.pred
merged_df['LABEL'] = (merged_df['LABEL_PROBABILITY'] > thr).astype(np.int32)

merged_df[['LABEL', 'LABEL_PROBABILITY']].to_csv(os.path.join(data_path, 'output', 'output.csv'), index=False)

print("True Pred", merged_df['LABEL'].sum(), merged_df['LABEL'].sum()/merged_df.shape[0])
print("Pred Prob", merged_df.LABEL_PROBABILITY.mean(), merged_df.LABEL_PROBABILITY.std())
print("Inference Done")
