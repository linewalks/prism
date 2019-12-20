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
  def __init__(self, data_path='/data/train'):
    self.data_path = data_path

    self.extract_from_file()

  def extract_from_file(self):

    # 각 테이블에서 필요한 정보만 남기고 정리
    # - 불필요 컬럼 제거
    # - outlier, null 값 처리 등
    person_df = self.extract_person()
    condition_df = self.extract_condition()
    measurement_df = self.extract_measurement()

    print(person_df.head())
    print(condition_df.head())
    print(measurement_df.head())

    # 시간대별로 Group

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

    # 필요 컬럼만 사용
    condition_df = condition_df[['PERSON_ID', 'CONDITION_SOURCE_VALUE', 'CONDITION_START_DATETIME']]
    return condition_df

  def extract_measurement(self):
    measurement_df = pd.read_csv(os.path.join(self.data_path, 'MEASUREMENT_NICU.csv'))

    source_value_invert_map = {}
    for new_value in MEASUREMENT_SOURCE_VALUE_MAP:
      for table_value in MEASUREMENT_SOURCE_VALUE_MAP[new_value]:
        source_value_invert_map[table_value] = new_value

    
    return measurement_df
