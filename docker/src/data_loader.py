import os
import time
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences

CONDITION_LIST = [
    'A49.0',
    'B34.9',
    'B95.6',
    'D18.0',
    'D22.5',
    'D22.9',
    'D64.9',
    'D75.8',
    'E03.1',
    'E03.8',
    'E03.9',
    'E22.2',
    'E27.4',
    'E55.0',
    'E63.9',
    'E80.7',
    'E83.5',
    'E87.1',
    'E87.2',
    'E88.9',
    'F19.0',
    'F98.2',
    'G00.2',
    'G00.8',
    'G25.3',
    'G91.8',
    'G93.8',
    'H35.1',
    'H65.9',
    'H66.9',
    'H90.2',
    'H91.9',
    'H93.2',
    'I27.2',
    'I28.8',
    'I47.1',
    'I50.9',
    'J06.9',
    'J12.3',
    'J18.9',
    'J21.1',
    'J21.9',
    'J40',
    'J93.9',
    'J98.1',
    'K21.9',
    'K40.3',
    'K40.9',
    'K56.5',
    'K59.0',
    'K61.0',
    'K91.4',
    'K92.2',
    'L05.9',
    'L22',
    'L74.3',
    'L92.8',
    'M43.62',
    'M67.4',
    'N13.3',
    'N17.9',
    'N43.3',
    'N48.1',
    'N83.2',
    'N94.8',
    'O31.2',
    'P00.0',
    'P00.8',
    'P00.9',
    'P01.0',
    'P01.1',
    'P01.2',
    'P01.3',
    'P01.5',
    'P01.7',
    'P02.0',
    'P02.1',
    'P02.4',
    'P02.7',
    'P03.4',
    'P04.0',
    'P05.1',
    'P05.9',
    'P07.0',
    'P07.1',
    'P07.2',
    'P07.3',
    'P08.1',
    'P22.0',
    'P22.9',
    'P25.1',
    'P26.1',
    'P26.9',
    'P27.1',
    'P28.2',
    'P28.4',
    'P29.3',
    'P29.8',
    'P35.1',
    'P52.0',
    'P52.1',
    'P52.2',
    'P52.3',
    'P52.6',
    'P52.8',
    'P52.9',
    'P54.0',
    'P54.3',
    'P56.9',
    'P59.0',
    'P59.8',
    'P59.9',
    'P61.0',
    'P61.2',
    'P70.0',
    'P70.4',
    'P72.2',
    'P74',
    'P76.0',
    'P78.1',
    'P81.9',
    'P90',
    'P91.2',
    'P91.7',
    'P92.9',
    'P94.2',
    'P96.8',
    'Q04.3',
    'Q04.6',
    'Q04.8',
    'Q10.3',
    'Q10.5',
    'Q21.1',
    'Q25.0',
    'Q27.0',
    'Q31.5',
    'Q33.6',
    'Q42.3',
    'Q53.1',
    'Q53.2',
    'Q53.9',
    'Q62.0',
    'Q63.2',
    'Q64.4',
    'Q65.2',
    'Q66.8',
    'Q67.3',
    'Q69.1',
    'Q82.8',
    'Q87.2',
    'Q89.9',
    'R00.1',
    'R04.8',
    'R05',
    'R09.2',
    'R09.3',
    'R09.8',
    'R11',
    'R16.2',
    'R21',
    'R25.1',
    'R34',
    'R49.0',
    'R50.9',
    'R56.8',
    'R62.0',
    'R62.8',
    'R68.1',
    'R73.9',
    'R94.6',
    'S09.9',
    'S36.4',
    'T18.9',
    'T81.3',
    'U83.0',
    'Z00.0',
    'Z01.0',
    'Z13.9',
    'Z26.9',
    'Z38.0',
    'Z38.1',
    'Z38.3',
    'Z38.6'
]

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
    self.condition_cols = None
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
    condition_df.CONDITION_START_DATETIME = pd.to_datetime(condition_df.CONDITION_START_DATETIME, utc=True)

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
    measurement_df.MEASUREMENT_DATETIME = pd.to_datetime(measurement_df.MEASUREMENT_DATETIME, utc=True)

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

    group_cols = ['PERSON_ID', 'CONDITION_START_DATETIME', 'CONDITION_SOURCE_VALUE']
    print(condition_df.CONDITION_SOURCE_VALUE.unique())
    condition_df = pd.pivot_table(condition_df, index=group_cols[:2],
                                  columns=group_cols[2], aggfunc=len, fill_value=0)

    condition_new_cols = {x: 0 for x in CONDITION_LIST if x not in condition_df.columns}
    condition_df = condition_df.assign(**condition_new_cols)
    print([x for x in condition_df.columns if x not in CONDITION_LIST])
    print("data_loader groupby_hour_condition time:", time.time() - start_time)
    return condition_df

  def groupby_hour_measurement(self, measurement_df):
    start_time = time.time()
    measurement_df['MEASUREMENT_DATE'] = measurement_df.MEASUREMENT_DATETIME.dt.date
    # timestamp로 join 하기 위하여 시간 포맷을 utc로 통일
    measurement_df['MEASUREMENT_DATE'] = pd.to_datetime(measurement_df['MEASUREMENT_DATE'], utc=True)

    measurement_df['MEASUREMENT_HOUR'] = measurement_df.MEASUREMENT_DATETIME.dt.hour
    measurement_df['MEASUREMENT_HOURGRP'] = measurement_df.MEASUREMENT_HOUR // self.group_hour

    group_cols = ['PERSON_ID', 'MEASUREMENT_DATE', 'MEASUREMENT_HOURGRP', 'MEASUREMENT_SOURCE_VALUE']
    measurement_df = measurement_df.groupby(group_cols) \
                                   .VALUE_AS_NUMBER.agg(['count', 'min', 'max', 'mean', 'std', 'var'])

    measurement_df = measurement_df.unstack().reset_index().fillna(0)
    new_cols = []
    for col in measurement_df.columns:
      if col[1] == '':
        new_cols.append(col[0])
      else:
        new_cols.append(col)
    measurement_df.columns = new_cols
    print("data_loader groupby_hour_measurement time:", time.time() - start_time)
    return measurement_df

  def make_data(self):
    start_time = time.time()
    # 빠른 서치를 위하여 데이터 정렬
    # 가장 마지막 시점이 먼저 오도록 반대로 정렬
    cohort_df = self.cohort_df.sort_values(['SUBJECT_ID', 'COHORT_END_DATE'], ascending=[True, False])
    # measurement_ary = self.measurement_df.sort_values(['PERSON_ID', 'MEASUREMENT_DATE', 'MEASUREMENT_HOURGRP'],
    #                                                   ascending=[True, False, False]).values

    # timestamp가 가장 세밀한 검진에 진단을 붙임
    feature_df = pd.merge(self.measurement_df, self.condition_df.reset_index(), how='left',
                          left_on=['PERSON_ID', 'MEASUREMENT_DATE'], right_on=['PERSON_ID', 'CONDITION_START_DATETIME'])
    feature_df = feature_df.drop('CONDITION_START_DATETIME', axis=1)
    # 새로운 진단이 나올때까지 직전의 진단을 유지
    feature_df[CONDITION_LIST] = feature_df[CONDITION_LIST].fillna(method='ffill')

    feature_df = feature_df.sort_values(['PERSON_ID', 'MEASUREMENT_DATE', 'MEASUREMENT_HOURGRP'],
                                        ascending=[True, False, False])
    feature_ary = feature_df.values

    cols = ['SUBJECT_ID', 'COHORT_END_DATE']
    if self.is_train:
      cols.append('LABEL')

    x_list = []
    y_list = []
    key_list = []
    feature_idx = 0

    for row in cohort_df[cols].values:
      subject_id = row[0]
      cohort_end_date = row[1]

      # key에 맞는 measurement를 찾는다
      while True:
        feature_row = feature_ary[feature_idx]

        person_id = feature_row[0]
        feature_date = feature_row[1]
        feature_hourgrp = feature_row[2]

        feature_datetime = datetime(feature_date.year,
                                    feature_date.month,
                                    feature_date.day,
                                    feature_hourgrp * self.group_hour)
        if person_id == subject_id and feature_datetime < cohort_end_date:
          # 같은 환자이고 cohort_end_date보다 먼저 발생한 데이터이면
          # 맞는 데이터
          each_x_list = []
          for timestep in range(self.timestep_per_data):
            if feature_idx + timestep >= len(feature_ary):
              break
            timestep_row = feature_ary[feature_idx + timestep]
            timestep_person_id = timestep_row[0]
            if timestep_person_id == subject_id:
              timestep_data = timestep_row[3:]
              each_x_list.append(timestep_data)
            else:
              break
          x_list.append(np.array(each_x_list))
          break
        elif person_id > subject_id:
          # 데이터를 못찾음. 다음 환자로 넘어가버렸다
          feature_data = feature_row[3:]
          x_list.append(np.array([[0] * len(feature_data)]))
          break
        else:
          # 탐색이 더 필요함
          feature_idx += 1

      # y 추가
      if self.is_train:
        label = row[2]
        y_list.append(label)

      key_list.append((row[0], row[1]))

    self.x = np.array(x_list)
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
