import os
import pandas as pd


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
  def __init__(self, data_path='/data/train', is_train=True, group_hour=1):
    self.data_path = data_path
    self.is_train = is_train

    self.group_hour = group_hour

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

  def extract_outcome_cohort(self):
    cohort_df = pd.read_csv(os.path.join(self.data_path, 'OUTCOME_COHORT.csv'))
    return cohort_df

  def extract_person(self):
    person_df = pd.read_csv(os.path.join(self.data_path, 'PERSON_NICU.csv'))
    person_df = pd.concat([
        person_df[['PERSON_ID', 'BIRTH_DATETIME']],
        pd.get_dummies(person_df.GENDER_SOURCE_VALUE, prefix='gender')
    ], axis=1)
    return person_df

  def extract_condition(self):
    condition_df = pd.read_csv(os.path.join(self.data_path, 'CONDITION_OCCURRENCE_NICU.csv'))
    # Null 이거나 값이 빈 것을 날림
    condition_df = condition_df[pd.notnull(condition_df.CONDITION_SOURCE_VALUE)]
    condition_df = condition_df[condition_df.CONDITION_SOURCE_VALUE.str.len() > 0]

    condition_df.CONDITION_START_DATETIME = pd.to_datetime(condition_df.CONDITION_START_DATETIME)

    # 필요 컬럼만 사용
    condition_df = condition_df[['PERSON_ID', 'CONDITION_SOURCE_VALUE', 'CONDITION_START_DATETIME']]
    return condition_df

  def extract_measurement(self):
    measurement_df = pd.read_csv(os.path.join(self.data_path, 'MEASUREMENT_NICU.csv'))

    # source_value 맵핑
    source_value_invert_map = {}
    for new_value in MEASUREMENT_SOURCE_VALUE_MAP:
      for table_value in MEASUREMENT_SOURCE_VALUE_MAP[new_value]:
        source_value_invert_map[table_value] = new_value
    measurement_df.MEASUREMENT_SOURCE_VALUE = measurement_df.MEASUREMENT_SOURCE_VALUE.replace(source_value_invert_map)

    measurement_df.MEASUREMENT_DATETIME = pd.to_datetime(measurement_df.MEASUREMENT_DATETIME)
    # 필요 컬럼만 사용
    measurement_df = measurement_df[['PERSON_ID', 'MEASUREMENT_DATETIME', 'MEASUREMENT_SOURCE_VALUE', 'VALUE_AS_NUMBER']]
    return measurement_df

  def groupby_hour(self):
    self.groupby_hour_condition()

  def groupby_hour_condition(self):
    pass
