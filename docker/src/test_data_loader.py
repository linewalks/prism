import os
import numpy as np
from data_loader import DataLoader

data_path = '../data/'
data_loader = DataLoader(data_path=os.path.join(data_path, 'train'), pytest=True)


class Test_DataLoader():

  def test_read_csv(self):
    data_loader.cohort_df = data_loader.extract_outcome_cohort()
    data_loader.person_df = data_loader.extract_person()
    data_loader.condition_df = data_loader.extract_condition()
    data_loader.measurement_df = data_loader.extract_measurement()

  def test_group_hour(self):
    data_loader.groupby_hour()

  def test_make_data(self):
    data_loader.make_data()
    assert data_loader.x[0][0].shape == (60,)
    assert data_loader.y.shape[0] == data_loader.x.shape[0]

  def test_split_data(self):
    data_loader.split_data()
    _, train_y = data_loader.get_train_data()
    _, valid_y = data_loader.get_valid_data()
    assert np.array_equal(np.unique(train_y), np.array([0, 1]))
    assert np.array_equal(np.unique(valid_y), np.array([0, 1]))
