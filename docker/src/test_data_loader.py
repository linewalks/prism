import os
import numpy as np
import pandas as pd
from data_loader import DataLoader

data_path = '../data/'
data_loader = DataLoader(data_path=os.path.join(data_path, 'train'),
                         common_path=os.path.join(data_path, 'volume'),
                         task_path=os.path.join(data_path, 'volume', 'local_test'),
                         pytest=True)


class Test_DataLoader():

  def test_read_csv(self):
    data_loader.cohort_df = data_loader.extract_outcome_cohort()
    data_loader.person_df = data_loader.extract_person()
    data_loader.condition_df = data_loader.extract_condition()
    data_loader.measurement_df = data_loader.extract_measurement()

  def test_group_hour(self):
    data_loader.groupby_hour()

  def test_make_data(self):
    data_loader.make_person_sequence()
    data_loader.make_data()
    assert data_loader.x[0][0].shape == (72,)
    assert data_loader.y.shape[0] == data_loader.x.shape[0]

  def test_split_data(self):
    data_loader.split_data()
    _, train_y = data_loader.get_train_data()
    _, valid_y = data_loader.get_valid_data()

    idx_true_label = np.where(data_loader.y == 1)[0]
    assert data_loader.y.sum() == data_loader.y[idx_true_label].sum()
    assert np.array_equal(np.unique(train_y), np.array([0, 1]))
    assert np.array_equal(np.unique(valid_y), np.array([0, 1]))

  def test_split_data_shuffle(self):
    n = 10
    n_true = 2
    data_loader.x = pd.DataFrame(np.arange(n), columns=['SUBJECT_ID'])
    data_loader.key = data_loader.x
    data_loader.y = np.concatenate([np.zeros(n - n_true), np.ones(n_true)])

    data_loader._stratified_shuffle()
    assert data_loader.train_y.sum() == 1
    assert data_loader.valid_y.sum() == 1

  def test_measurement_clip(self):
    hr = [0, 130, 150, 183, 190]
    rr = [0, 27, 50, 70, 80]
    rows = [['HR', v] for v in hr] + [['RR', v] for v in rr]

    df = pd.DataFrame(rows, columns=['MEASUREMENT_SOURCE_VALUE', 'VALUE_AS_NUMBER'])
    print(df)
    df.VALUE_AS_NUMBER = df.apply(lambda row:
                                  data_loader._clip_measurement(
                                      row['MEASUREMENT_SOURCE_VALUE'],
                                      row['VALUE_AS_NUMBER']),
                                  axis=1)

