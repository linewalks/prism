import os
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
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
  def __init__(self, data_path='/data/train', common_path='/data/volume',
               task_path='/data/volume/local_test',
               is_train=True,
               group_hour=1, timestep_per_data=128,
               measurement_normalize=True,
               valid_size=0.2, data_split_random_seed=1235, pytest=False):
    self.data_path = data_path
    self.common_path = common_path
    self.task_path = task_path
    self.is_train = is_train

    self.group_hour = group_hour
    self.timestep_per_data = timestep_per_data

    self.measurement_normalize = measurement_normalize

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

    # 환자별 시간 시퀀스 데이터를 만듦
    self.make_person_sequence()

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

    # source_value별 평균값 추출
    if self.is_train:
      self.measurement_mean_df = measurement_df.groupby('MEASUREMENT_SOURCE_VALUE').VALUE_AS_NUMBER.mean()
      self.measurement_mean_df.to_pickle(os.path.join(self.common_path, 'measurement_mean.pkl'))
    else:
      # inference일 경우 저장된 걸 불러온다
      self.measurement_mean_df = pd.read_pickle(os.path.join(self.common_path, 'measurement_mean.pkl'))
    
    print("data_loader extract_measurement time:", time.time() - start_time)
    return measurement_df

  def groupby_hour(self):
    self.condition_df = self.groupby_hour_condition(self.condition_df)
    self.measurement_df = self.groupby_hour_measurement(self.measurement_df)

  def groupby_hour_condition(self, condition_df):
    start_time = time.time()

    condition_df['CONDITION_DATE'] = condition_df.CONDITION_START_DATETIME.dt.date
    condition_df['CONDITION_DATE'] = pd.to_datetime(condition_df.CONDITION_DATE, utc=True)

    # 진단은 시간이 없다. 당일의 마지막에 진단 받은걸로 가정한다
    condition_df['HOURGRP'] = 23 // self.group_hour

    group_cols = ['PERSON_ID', 'CONDITION_DATE', 'HOURGRP', 'CONDITION_SOURCE_VALUE']
    self.condition_cols = condition_df.CONDITION_SOURCE_VALUE.unique().tolist()

    condition_df['DUMMY'] = condition_df['CONDITION_SOURCE_VALUE']
    condition_df = condition_df.groupby(group_cols) \
                               .DUMMY.count().unstack().reset_index().fillna(0)

    condition_df = condition_df.rename(columns={'CONDITION_DATE': 'DATE'})

    condition_col_filename = os.path.join(self.task_path, 'condition_cols.npy')
    if self.is_train:
      # 컬럼 이름 저장
      np.save(condition_col_filename, np.array(condition_df.columns))
    else:
      # 컬럼 로드
      condition_cols = np.load(condition_col_filename, allow_pickle=True)
      new_condition_list = []
      for col in condition_cols:
        if col in condition_df.columns:
          new_condition_list.append(condition_df[col])
        else:
          new_condition_list.append(pd.Series([0] * condition_df.shape[0]))

      condition_df = pd.concat(new_condition_list, axis=1)
      condition_df.columns = condition_cols
    print("data_loader groupby_hour_condition time:", time.time() - start_time)
    return condition_df

  def groupby_hour_measurement(self, measurement_df):
    start_time = time.time()
    # timestamp로 join 하기 위하여 시간 포맷을 utc로 통일
    measurement_df['MEASUREMENT_DATE'] = measurement_df.MEASUREMENT_DATETIME.dt.date
    measurement_df['MEASUREMENT_DATE'] = pd.to_datetime(measurement_df.MEASUREMENT_DATE, utc=True)

    measurement_df['MEASUREMENT_HOUR'] = measurement_df.MEASUREMENT_DATETIME.dt.hour
    measurement_df['MEASUREMENT_HOURGRP'] = measurement_df.MEASUREMENT_HOUR // self.group_hour

    # 평균값 이용하여 Normalize
    if self.measurement_normalize: 
      measurement_df = pd.merge(measurement_df, 
                                self.measurement_mean_df.reset_index().rename(columns={'VALUE_AS_NUMBER': 'MEAN_VALUE'}),
                                on='MEASUREMENT_SOURCE_VALUE', how='left')
      measurement_df.VALUE_AS_NUMBER = measurement_df.VALUE_AS_NUMBER / measurement_df.MEAN_VALUE
    
    group_cols = ['PERSON_ID', 'MEASUREMENT_DATE', 'MEASUREMENT_HOURGRP', 'MEASUREMENT_SOURCE_VALUE']
    agg_list = ['count', 'min', 'max', 'mean', 'std', 'var']
    measurement_df = measurement_df.groupby(group_cols) \
                                   .VALUE_AS_NUMBER.agg(agg_list)

    measurement_df = measurement_df.unstack().reset_index().fillna(0)

    # 컬럼 이름 정제 (그룹화 하기 쉽게)
    new_cols = []
    for col in  measurement_df.columns:
      if col[1] == '':
        new_cols.append(col[0])
      elif col[0] in agg_list:
        new_cols.append((col[1], col[0]))
    measurement_df.columns = new_cols

    measurement_df = measurement_df.rename(columns={'MEASUREMENT_DATE': 'DATE',
                                                    'MEASUREMENT_HOURGRP': 'HOURGRP'})

    measurement_col_filename = os.path.join(self.task_path, 'measurement_cols.npy')
    if self.is_train:
      # 컬럼 이름 저장
      np.save(measurement_col_filename, np.array(measurement_df.columns))
    else:
      # 컬럼 로드
      measurement_cols = np.load(measurement_col_filename, allow_pickle=True)
      new_measurement_list = []
      for col in measurement_cols:
        if col in measurement_df.columns:
          new_measurement_list.append(measurement_df[col])
        else:
          new_measurement_list.append(pd.Series([0] * measurement_df.shape[0]))

      measurement_df = pd.concat(new_measurement_list, axis=1)
      measurement_df.columns = measurement_cols
    print("data_loader groupby_hour_measurement time:", time.time() - start_time)
    return measurement_df

  def make_person_sequence(self):
    start_time = time.time()
    # 환자별로 데이터의 시작시간과 종료시간을 구한다.
    timerange_df = self.cohort_df.groupby('SUBJECT_ID').agg({'COHORT_START_DATE': 'min', 'COHORT_END_DATE': 'max'})
    timerange_df['START_DATE'] = timerange_df.COHORT_START_DATE.dt.date
    timerange_df['START_HOURGRP'] = timerange_df.COHORT_START_DATE.dt.hour // self.group_hour
    timerange_df['END_DATE'] = timerange_df.COHORT_END_DATE.dt.date
    timerange_df['END_HOURGRP'] = timerange_df.COHORT_END_DATE.dt.hour // self.group_hour
    timerange_df = timerange_df.drop(['COHORT_START_DATE', 'COHORT_END_DATE'], axis=1)

    condition_ary = self.condition_df.sort_values(['PERSON_ID', 'DATE', 'HOURGRP'], ascending=True).values
    measurement_ary = self.measurement_df.sort_values(['PERSON_ID', 'DATE', 'HOURGRP'], ascending=True).values
    timerange_ary = timerange_df.sort_values('SUBJECT_ID', ascending=True).reset_index().values

    condition_cols = self.condition_df.columns[3:]
    measurement_cols = self.measurement_df.columns[3:]

    # 빈 Time Range 없게 시간대 정보를 채움
    max_hourgrp = (24 // self.group_hour) - 1
    
    key_list = []
    for person_id, start_date, start_hourgrp, end_date, end_hourgrp in timerange_ary:
      cur_date = start_date
      cur_hourgrp = start_hourgrp
      
      while True:
        key_list.append((person_id, cur_date, cur_hourgrp))

        cur_hourgrp += 1                  # 1 그룹시간만큼 탐색
        if cur_hourgrp > max_hourgrp:     # 다음 날짜로 넘어감
          cur_date = cur_date + timedelta(days=1)
          cur_hourgrp = 0

        if cur_date > end_date or \
           (cur_date == end_date and cur_hourgrp >= end_hourgrp):
           # 끝까지 탐색함
           break

    # 시간대 정보에 따라 데이터를 채워 넣는다
    condition_idx = measurement_idx = 0
    prev_person_id = None
    prev_conditions = None

    data_list = []
    for person_id, date, hourgrp in key_list:
      data = [person_id, date, hourgrp]
      
      # Measurement 탐색
      while True:
        if measurement_idx >= len(measurement_ary):
          data.extend([0] * len(measurement_data))
          break
          
        measurement_row = measurement_ary[measurement_idx]
        measurement_person_id = measurement_row[0]
        measurement_date = measurement_row[1]
        measurement_hourgrp = measurement_row[2]
        measurement_data = measurement_row[3:]

        state = 0       # 0: 다음 데이터 탐색 1: 맞는 데이터 찾음 2: 맞는 데이터 없음
        if measurement_person_id > person_id:       # 다음 환자로 넘어감
          state = 2
        elif measurement_person_id == person_id:
          if measurement_date > date:               # 다음 날짜로 넘어감
            state = 2
          elif measurement_date == date:
            if measurement_hourgrp > hourgrp:       # 다음 그룹시간으로 넘어감
              state = 2
            elif measurement_hourgrp == hourgrp:    # 맞는 데이터
              state = 1

        if state == 0:                  # 계속 탐색
          measurement_idx += 1
        elif state == 1:                # 데이터 찾음
          data.extend(measurement_data)
          measurement_idx += 1
          break
        elif state == 2:                # 맞는 데이터가 없음
          data.extend([0] * len(measurement_data))
          break

      # Condition 탐색
      # 이전과 다른 환자임. condition정보 reset
      if prev_person_id != person_id:
        prev_conditions = np.array([0] * len(condition_ary[0][3:]))
      
      while True:
        if condition_idx >= len(condition_ary):
          data.extend(prev_conditions)
          break

        condition_row = condition_ary[condition_idx]
        condition_person_id = condition_row[0]
        condition_date = condition_row[1]
        condition_hourgrp = condition_row[2]
        condition_data = condition_row[3:]

        state = 0       # 0: 다음 데이터 탐색 1: 맞는 데이터 찾음 2: 맞는 데이터 없음
        if condition_person_id > person_id:       # 다음 환자로 넘어감
          state = 2
        elif condition_person_id == person_id:
          if condition_date > date:               # 다음 날짜로 넘어감
            state = 2
          elif condition_date == date:
            if condition_hourgrp > hourgrp:       # 다음 그룹시간으로 넘어감
              state = 2
            elif condition_hourgrp == hourgrp:    # 맞는 데이터
              state = 1

        if state == 0:                  # 계속 탐색
          condition_idx += 1
        elif state == 1:                # 데이터 찾음
          # 이전 Condition 정보와 나중 Condition 정보를 합친다
          prev_conditions = np.array(prev_conditions) + np.array(condition_data)
          data.extend(prev_conditions)
          condition_idx += 1
          break
        elif state == 2:                # 맞는 데이터가 없음
          data.extend(prev_conditions)
          break

      data_list.append(data)
      prev_person_id = person_id

    self.feature_df = pd.DataFrame(data_list, 
                                   columns=['PERSON_ID', 'DATE', 'HOURGRP'] + list(measurement_cols) + list(condition_cols))
    print(self.feature_df)
    print("data_loader make_person_sequence time:", time.time() - start_time)

  def make_data(self):
    start_time = time.time()
    # 빠른 서치를 위하여 데이터 정렬
    # 가장 마지막 시점이 먼저 오도록 반대로 정렬
    cohort_df = self.cohort_df.sort_values(['SUBJECT_ID', 'COHORT_END_DATE'], ascending=[True, False])
    feature_ary = self.feature_df.sort_values(['PERSON_ID', 'DATE', 'HOURGRP'], ascending=[True, False, False]).values
    
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

      # key에 맞는 data feature 찾는다
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
          # 가장 나중 데이터부터 each_x_list에 넣었으니 데이터에 넣을땐 반대로
          x_list.append(np.array(each_x_list)[::-1])
          break
        elif person_id > subject_id:
          # 데이터를 못찾음. 다음 환자로 넘어가버렸다
          print("Person's data not found", subject_id)
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
