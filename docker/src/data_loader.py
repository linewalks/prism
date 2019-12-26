import os
import time
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences


MEASUREMENT_SOURCE_VALUE_MAP = {
    "IDBP": ["ARTd", "ABPd"],
    "IMBP": ["ABPm"],
    "ISBP": ["ARTs", "ABPs"],
    "FDBP": ["AoD"],
    "FMBP": ["AoS"],
    "FSBP": ["AoS"],
    "BT": ["Tskin", "Trect", "Tnaso", "Tesoph", "Temp", "Tcore"],
    "CVP": ["CVPm"],
    "ETCO2": ["etCO2"],
    "PR": ["HR", "Pulse"],
    "LAP": ["LAPm"],
    "MINUTE_VOLUME": ["MINVOL", "MV"],
    "PMEAN": ["MnAwP", "Pmean"],
    "DBP": ["NBPd", "NBP-D"],
    "MBP": ["NBPm", "NBP-M"],
    "SBP": ["NBPs", "NBP-S"],
    "DPAP": ["PAPd", "Pd"],
    "MPAP": ["PAPm", "Pm"],
    "SPAP": ["PAPs", "Ps"],
    "PPEAK": ["PIP", "Ppeak"],
    "RR": ["RR, Resp"],
    "FREQ_MEASURE": ["RRaw"],
    "SPO2": ["SpO2T", "SpO2-%", "SpO2"],
    "VTE": ["TV"],
    "VIT": ["TVin"]
}


class DataLoader:
  def __init__(self, data_path='/data/train', is_train=True,
               group_hour=1, timestep_per_data=128,
               valid_size=0.2, data_split_random_seed=1234, pytest=False):
    self.data_path = data_path
    self.is_train = is_train

    self.group_hour = group_hour
    self.timestep_per_data = timestep_per_data

    self.valid_size = valid_size
    self.data_split_random_seed = data_split_random_seed
    if not pytest:
      self.extract_from_file()

  def extract_from_file(self):

    # 각 테이블에서 필요한 정보만 남기고 정리
    # - 불필요 컬럼 제거
    # - outlier, null 값 처리 등
    self.cohort_df = self.extract_outcome_cohort()
    self.person_df = self.extract_person()
    self.condition_df = self.extract_condition()
    self.measurement_df = self.extract_measurement()

    # 데이터를 시간대별로 Group
    self.groupby_hour()

    # cohort_df에 맞추어 모델에 들어갈 데이터를 만듦
    self.make_data()

    # 환자 기준으로 train/valid split
    self.split_data()

  def extract_outcome_cohort(self):
    start_time = time.time()
    cohort_df = pd.read_csv(os.path.join(self.data_path, 'OUTCOME_COHORT.csv'), encoding='windows-1252')

    cohort_df.COHORT_START_DATE = pd.to_datetime(cohort_df.COHORT_START_DATE)
    cohort_df.COHORT_END_DATE = pd.to_datetime(cohort_df.COHORT_END_DATE)
    print("data_loader extract_outcome_cohort time:", time.time() - start_time)
    return cohort_df

  def extract_person(self):
    start_time = time.time()
    person_df = pd.read_csv(os.path.join(self.data_path, 'PERSON_NICU.csv'), encoding='windows-1252')
    person_df = pd.concat([
        person_df[['PERSON_ID', 'BIRTH_DATETIME']],
        pd.get_dummies(person_df.GENDER_SOURCE_VALUE, prefix='gender')
    ], axis=1)
    print("data_loader extract_person time:", time.time() - start_time)
    return person_df

  def extract_condition(self):
    start_time = time.time()
    condition_df = pd.read_csv(os.path.join(self.data_path, 'CONDITION_OCCURRENCE_NICU.csv'), encoding='windows-1252')
    # Null 이거나 값이 빈 것을 날림
    condition_df = condition_df[pd.notnull(condition_df.CONDITION_SOURCE_VALUE)]
    condition_df = condition_df[condition_df.CONDITION_SOURCE_VALUE.str.len() > 0]

    # 컬럼 타입 설정
    condition_df.CONDITION_START_DATETIME = pd.to_datetime(condition_df.CONDITION_START_DATETIME)

    # 필요 컬럼만 사용
    condition_df = condition_df[['PERSON_ID', 'CONDITION_SOURCE_VALUE', 'CONDITION_START_DATETIME']]
    print("data_loader extract_condition time:", time.time() - start_time)
    return condition_df

  def extract_measurement(self):
    start_time = time.time()
    measurement_df = pd.read_csv(os.path.join(self.data_path, 'MEASUREMENT_NICU.csv'), encoding='windows-1252')

    # source_value 맵핑
    source_value_invert_map = {}
    for new_value in MEASUREMENT_SOURCE_VALUE_MAP:
      for table_value in MEASUREMENT_SOURCE_VALUE_MAP[new_value]:
        source_value_invert_map[table_value] = new_value
    measurement_df.MEASUREMENT_SOURCE_VALUE = measurement_df.MEASUREMENT_SOURCE_VALUE.replace(source_value_invert_map)

    # 맵핑이된 정보만 남긴다
    measurement_df = measurement_df[measurement_df.MEASUREMENT_SOURCE_VALUE.isin(MEASUREMENT_SOURCE_VALUE_MAP.keys())]

    # 컬럼 타입 설정
    measurement_df.MEASUREMENT_DATETIME = pd.to_datetime(measurement_df.MEASUREMENT_DATETIME)

    # 필요 컬럼만 사용
    measurement_df = measurement_df[['PERSON_ID', 'MEASUREMENT_DATETIME',
                                     'MEASUREMENT_SOURCE_VALUE', 'VALUE_AS_NUMBER']]
    print("data_loader extract_measurement time:", time.time() - start_time)
    return measurement_df

  def groupby_hour(self):
    self.condition_df = self.groupby_hour_condition(self.condition_df)
    self.measurement_df = self.groupby_hour_measurement(self.measurement_df)

  def groupby_hour_condition(self, condition_df):
    start_time = time.time()
    # condition_df["CONDITION_DATE"] = condition_df.CONDITION_START_DATETIME.dt.date
    # condition_df["CONDITION_HOUR"] = condition_df.CONDITION_START_DATETIME.dt.hour
    group_cols = ['PERSON_ID', 'CONDITION_START_DATETIME']
    condition_df = condition_df.groupby(group_cols) \
        .CONDITION_SOURCE_VALUE.agg([len])
    condition_df = condition_df.unstack().reset_index().fillna(0)
    print(condition_df.head())
    print("data_loader groupby_hour_condition time:", time.time() - start_time)
    return condition_df

  def groupby_hour_measurement(self, measurement_df):
    start_time = time.time()
    measurement_df['MEASUREMENT_DATE'] = measurement_df.MEASUREMENT_DATETIME.dt.date
    measurement_df['MEASUREMENT_HOUR'] = measurement_df.MEASUREMENT_DATETIME.dt.hour
    measurement_df['MEASUREMENT_HOURGRP'] = measurement_df.MEASUREMENT_HOUR // self.group_hour

    group_cols = ['PERSON_ID', 'MEASUREMENT_DATE', 'MEASUREMENT_HOURGRP', 'MEASUREMENT_SOURCE_VALUE']
    measurement_df = measurement_df.groupby(group_cols) \
                                   .VALUE_AS_NUMBER.agg(['count', 'min', 'max', 'mean', 'std', 'var'])

    measurement_df = measurement_df.unstack().reset_index().fillna(0)
    print("data_loader groupby_hour_measurement time:", time.time() - start_time)
    return measurement_df

  def make_data(self):
    start_time = time.time()
    # 빠른 서치를 위하여 데이터 정렬
    # 가장 마지막 시점이 먼저 오도록 반대로 정렬
    cohort_df = self.cohort_df.sort_values(['SUBJECT_ID', 'COHORT_END_DATE'], ascending=[True, False])
    measurement_ary = self.measurement_df.sort_values(['PERSON_ID', 'MEASUREMENT_DATE', 'MEASUREMENT_HOURGRP'],
                                                      ascending=[True, False, False]).values
    condition_ary = self.condition_df.sort_values(['PERSON_ID', 'CONDITION_START_DATETIME'],
                                                  ascending=[True, False]).values

    cols = ['SUBJECT_ID', 'COHORT_END_DATE']
    if self.is_train:
      cols.append('LABEL')

    x_list_1 = []
    x_list_2 = []
    y_list = []
    key_list = []
    measurement_idx = 0
    condition_idx = 0

    for row in cohort_df[cols].values:
      subject_id = row[0]
      cohort_end_date = row[1]

      # key에 맞는 measurement를 찾는다
      while True:
        measurement_row = measurement_ary[measurement_idx]

        person_id = measurement_row[0]
        measurement_date = measurement_row[1]
        measurement_hourgrp = measurement_row[2]

        measurement_datetime = datetime(measurement_date.year,
                                        measurement_date.month,
                                        measurement_date.day,
                                        measurement_hourgrp * self.group_hour)
        if person_id == subject_id and measurement_datetime < cohort_end_date:
          # 같은 환자이고 cohort_end_date보다 먼저 발생한 데이터이면
          # 맞는 데이터
          each_x_list = []
          for timestep in range(self.timestep_per_data):
            if measurement_idx + timestep >= len(measurement_ary):
              break
            timestep_row = measurement_ary[measurement_idx + timestep]
            timestep_person_id = timestep_row[0]
            if timestep_person_id == subject_id:
              timestep_data = timestep_row[3:]
              each_x_list.append(timestep_data)
            else:
              break
          x_list_1.append(np.array(each_x_list))
          break
        elif person_id > subject_id:
          # 데이터를 못찾음. 다음 환자로 넘어가버렸다
          measurement_data = measurement_row[3:]
          x_list_1.append(np.array([[0] * len(measurement_data)]))
          break
        else:
          # 탐색이 더 필요함
          measurement_idx += 1

      while True:
        condition_row = condition_ary[condition_idx]
        person_id = condition_row[0]
        condition_date = condition_row[1]
        if person_id == subject_id and condition_date < cohort_end_date:
          # 같은 환자이고 cohort_end_date보다 먼저 발생한 데이터이면
          # 맞는 데이터
          each_x_list = []
          for timestep in range(self.timestep_per_data):
            if condition_idx + timestep >= len(condition_ary):
              break
            timestep_row = condition_ary[condition_idx + timestep]
            timestep_person_id = timestep_row[0]
            if timestep_person_id == subject_id:
              timestep_data = timestep_row[3:]
              each_x_list.append(timestep_data)
            else:
              break
          x_list_2.append(np.array(each_x_list))
          break
        elif person_id > subject_id:
          # 데이터를 못찾음. 다음 환자로 넘어가버렸다
          condition_data = condition_row[3:]
          x_list_2.append(np.array([[0] * len(condition_data)]))
          break
        else:
          # 탐색이 더 필요함
          condition_idx += 1

      # y 추가
      if self.is_train:
        label = row[2]
        y_list.append(label)

      key_list.append((row[0], row[1]))

    # self.x = np.concatenate((np.array(x_list_1), np.array(x_list_2)), axis=1)
    self.x = np.array(x_list_1)
    print(self.x.shape)
    self.y = np.array(y_list) if self.is_train else None
    self.key = pd.DataFrame(key_list, columns=['SUBJECT_ID', 'COHORT_END_DATE'])

    print("data_loader make_data time:", time.time() - start_time)

  def split_data(self):
    start_time = time.time()
    if self.is_train:
      try:
        train_patient, valid_patient = train_test_split(self.key.SUBJECT_ID.unique(),
                                                        test_size=self.valid_size,
                                                        random_state=self.data_split_random_seed)

        self.train_x = self.x[self.key.SUBJECT_ID.isin(train_patient)]
        self.train_y = self.y[self.key.SUBJECT_ID.isin(train_patient)]

        self.valid_x = self.x[self.key.SUBJECT_ID.isin(valid_patient)]
        self.valid_y = self.y[self.key.SUBJECT_ID.isin(valid_patient)]
      except ValueError:                                      # is sample data
        self.train_x = self.x
        self.train_y = self.y

        self.valid_x = self.x
        self.valid_y = self.y

      self.train_x = pad_sequences(self.train_x)
      self.valid_x = pad_sequences(self.valid_x)
    else:
      self.train_x = pad_sequences(self.x)

    print("data_loader split_data time:", time.time() - start_time)

  def get_train_data(self):
    return self.train_x, self.train_y

  def get_valid_data(self):
    return self.valid_x, self.valid_y

  def get_infer_data(self):
    return self.train_x
