import os
import time
import numpy as np
import pandas as pd
import pytz
import sys
from measurement_stat import MEASUREMENT_SOURCE_VALUE_STATS
from datetime import datetime, timedelta, time as datetime_time, timezone
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MinMaxScaler
from model import Autoencoder


MEASUREMENT_SOURCE_VALUE_MAP = {
    "IDBP": ["ABPd"],
    "IMBP": ["ABPm"],
    "ISBP": ["ABPs"],
    "FDBP": ["AoD"],
    "FMBP": ["AoM"],
    "FSBP": ["AoS"],
    "BT": ["Temp"],
    "CVP": ["CVPm"],
    "ETCO2": ["etCO2"],
    "PR": ["HR", "Pulse"],
    "MINVOL": ["MINVOL"],
    "MV": ["MV"],
    "PMEAN": ["MnAwP"],
    "DBP": ["NBPd"],
    "MBP": ["NBPm"],
    "SBP": ["NBPs"],
    "MPAP": ["PAPm"],
    "SPAP": ["PAPs", "Ps"],
    "PPEAK": ["PIP", "Ppeak"],
    "RR": ["RR, Resp"],
    "Rinsp": ["Rinsp"],
    "Rexp": ["Rexp"],
    "FREQ_MEASURE": ["RRaw"],
    "SPO2": ["SpO2T", "SpO2-%", "SpO2", "SPO2-r"],
    "VTE": ["TV"],
    "VIT": ["TVin"],
    "PEEP": ["PEEP"],
    'ST-I': ['ST-I'],
    'ST-II': ['ST-II'],
    'ST-III': ['ST-III'],
    'sTrig': ['sTrig'],
    'sTV': ['sTV'],
    'ST-V': ['ST-V'],
    'ST-V1': ['ST-V1'],
    'ST-V2': ['ST-V2'],
    'ST-V3': ['ST-V3'],
    'ST-V4': ['ST-V4'],
    'ST-V5': ['ST-V5'],
    'ST-V6': ['ST-V6']
}


MEASUREMENT_NORMALIZATION = ['mean', 'predefined']

dev_data = os.environ.get('DEV_DATA')
if dev_data is None:
    dev_data = ''
csv_files = {
    'person': f'{dev_data}PERSON_NICU.csv',
    'condition': f'{dev_data}CONDITION_OCCURRENCE_NICU.csv',
    'measurement': f'{dev_data}MEASUREMENT_NICU.csv',
    'outcome': f'{dev_data}OUTCOME_COHORT.csv'
}


class DataLoader:
    def __init__(self, data_path='/data/train', common_path='/data/volume',
                 task_path='/data/volume/local_test',
                 is_train=True,
                 group_hour=1, timestep_per_data=128,
                 measurement_normalize='mean',
                 condition_min_limit=0, condition_group=False,
                 autoencoder=False, predicted_set = [],
                 valid_size=0.2, data_split_random_seed=1235, pytest=False):
        self.data_path = data_path
        self.common_path = common_path
        self.task_path = task_path
        self.is_train = is_train

        self.group_hour = group_hour
        self.timestep_per_data = timestep_per_data

        assert measurement_normalize in MEASUREMENT_NORMALIZATION
        self.measurement_normalize = measurement_normalize

        self.condition_min_limit = condition_min_limit
        self.condition_group = condition_group
        self.autoencoder = autoencoder
        self.predicted_set = predicted_set
        self.valid_size = valid_size
        self.data_split_random_seed = data_split_random_seed
        if not pytest:
            self.extract_from_file()

    def extract_from_file(self):

        print('Load files', csv_files)
        # 각 테이블에서 필요한 정보만 남기고 정리
        # - 불필요 컬럼 제거
        # - outlier, null 값 처리 등
        self.cohort_df = self.extract_outcome_cohort()
        self.person_df = self.extract_person()
        self.condition_df = self.extract_condition()
        self.measurement_df = self.extract_measurement()
        # 데이터를 시간대별로 Group
        self.groupby_hour()
        if not self.autoencoder:


            # 환자별 시간 시퀀스 데이터를 만듦
            self.make_person_sequence()

            # 메모리 확보를 위하여 삭제
            del self.person_df
            del self.condition_df
            del self.measurement_df

            # cohort_df에 맞추어 모델에 들어갈 데이터를 만듦
            self.make_data()

            # 환자 기준으로 train/valid split
            self.split_data()

            del self.x
            del self.y

    def extract_outcome_cohort(self):
        start_time = time.time()
        cohort_df = pd.read_csv(os.path.join(
            self.data_path, csv_files['outcome']), encoding='windows-1252')

        cohort_df.COHORT_START_DATE = pd.to_datetime(
            cohort_df.COHORT_START_DATE)
        cohort_df.COHORT_END_DATE = pd.to_datetime(cohort_df.COHORT_END_DATE)
        print("data_loader extract_outcome_cohort time:",
              time.time() - start_time)
        return cohort_df

    def extract_person(self):
        start_time = time.time()
        person_df = pd.read_csv(os.path.join(
            self.data_path, csv_files['person']), encoding='windows-1252')
        person_df = pd.concat([
            person_df[['PERSON_ID', 'BIRTH_DATETIME']],
            pd.get_dummies(person_df.GENDER_SOURCE_VALUE, prefix='gender')
        ], axis=1)

        # 생일 컬럼 타입 설정
        person_df.BIRTH_DATETIME = pd.to_datetime(
            person_df.BIRTH_DATETIME, utc=True)
        # 여성/남성 컬럼 1개로 처리
        person_df.rename(columns={'gender_M': 'GENDER'}, inplace=True)
        if 'gender_F' in person_df.columns:
            del person_df['gender_F']

        print("data_loader extract_person time:", time.time() - start_time)
        return person_df

    def extract_condition(self):
        start_time = time.time()
        condition_df = pd.read_csv(os.path.join(self.data_path, csv_files['condition']), encoding='windows-1252',
                                   usecols=['PERSON_ID', 'CONDITION_SOURCE_VALUE', 'CONDITION_START_DATETIME'])
        # Null 이거나 값이 빈 것을 날림
        condition_df = condition_df[pd.notnull(
            condition_df.CONDITION_SOURCE_VALUE)]
        condition_df = condition_df[condition_df.CONDITION_SOURCE_VALUE.str.len(
        ) > 0]

        if self.condition_group:
            condition_df.CONDITION_SOURCE_VALUE = condition_df.CONDITION_SOURCE_VALUE.str.slice(
                stop=3)

        # 컬럼 타입 설정
        condition_df.CONDITION_START_DATETIME = pd.to_datetime(
            condition_df.CONDITION_START_DATETIME, utc=True)

        print("data_loader extract_condition time:", time.time() - start_time)
        return condition_df

    def extract_measurement(self):
        start_time = time.time()
        measurement_df = pd.read_csv(os.path.join(self.data_path, csv_files['measurement']),
                                     encoding='windows-1252',
                                     usecols=['PERSON_ID', 'MEASUREMENT_DATETIME',
                                              'MEASUREMENT_SOURCE_VALUE', 'VALUE_AS_NUMBER']
                                     )
        if self.measurement_normalize == MEASUREMENT_NORMALIZATION[0]:
            # source_value 맵핑
            source_value_invert_map = {}
            for new_value in MEASUREMENT_SOURCE_VALUE_MAP:
                for table_value in MEASUREMENT_SOURCE_VALUE_MAP[new_value]:
                    source_value_invert_map[table_value] = new_value
            measurement_df.MEASUREMENT_SOURCE_VALUE = measurement_df.MEASUREMENT_SOURCE_VALUE.replace(
                source_value_invert_map)

            # 맵핑이된 정보만 남긴다
            measurement_df = measurement_df[measurement_df.MEASUREMENT_SOURCE_VALUE.isin(
                MEASUREMENT_SOURCE_VALUE_MAP.keys())]

        # 컬럼 타입 설정
        measurement_df.MEASUREMENT_DATETIME = pd.to_datetime(
            measurement_df.MEASUREMENT_DATETIME, utc=True)

        # source_value별 평균값 추출
        if self.is_train:
            self.measurement_mean_df = measurement_df.groupby(
                'MEASUREMENT_SOURCE_VALUE').VALUE_AS_NUMBER.mean()
            self.measurement_mean_df.to_pickle(os.path.join(
                self.common_path, 'measurement_mean.pkl'))
        else:
            # inference일 경우 저장된 걸 불러온다
            self.measurement_mean_df = pd.read_pickle(
                os.path.join(self.common_path, 'measurement_mean.pkl'))

        print("data_loader extract_measurement time:", time.time() - start_time)
        return measurement_df

    def groupby_hour(self):
        self.condition_df = self.groupby_hour_condition(self.condition_df)
        self.measurement_df = self.groupby_hour_measurement(
            self.measurement_df)

    def groupby_hour_condition(self, condition_df):
        start_time = time.time()

        condition_df['CONDITION_DATE'] = condition_df.CONDITION_START_DATETIME.dt.date
        condition_df['CONDITION_DATE'] = pd.to_datetime(
            condition_df.CONDITION_DATE, utc=True)

        if self.is_train and self.condition_min_limit > 0:
            condition_group = condition_df.groupby(
                'CONDITION_SOURCE_VALUE').PERSON_ID.count()
            condition_group = condition_group[condition_group >
                                              self.condition_min_limit].index

            condition_df = condition_df[condition_df.CONDITION_SOURCE_VALUE.isin(
                condition_group)]

        # 진단은 시간이 없다. 당일의 마지막에 진단 받은걸로 가정한다
        condition_df['HOURGRP'] = 23 // self.group_hour

        group_cols = ['PERSON_ID', 'CONDITION_DATE',
                      'HOURGRP', 'CONDITION_SOURCE_VALUE']

        condition_df['DUMMY'] = condition_df['CONDITION_SOURCE_VALUE']
        condition_df = condition_df.groupby(group_cols) \
            .DUMMY.count().unstack().reset_index().fillna(0)

        condition_df = condition_df.rename(columns={'CONDITION_DATE': 'DATE'})

        condition_col_filename = os.path.join(
            self.task_path, 'condition_cols.npy')
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
                    new_condition_list.append(
                        pd.Series([0] * condition_df.shape[0]))

            condition_df = pd.concat(new_condition_list, axis=1)
            condition_df.columns = condition_cols
        print("data_loader groupby_hour_condition time:",
              time.time() - start_time)
        return condition_df

    def _clip_measurement(self, measurement_source_value, value_as_number):
        if value_as_number > MEASUREMENT_SOURCE_VALUE_STATS[measurement_source_value]['95%']:
            value_as_number = MEASUREMENT_SOURCE_VALUE_STATS[measurement_source_value]['95%']
        elif value_as_number < MEASUREMENT_SOURCE_VALUE_STATS[measurement_source_value]['5%']:
            value_as_number = MEASUREMENT_SOURCE_VALUE_STATS[measurement_source_value]['5%']
        return value_as_number

    def groupby_hour_measurement(self, measurement_df):
        start_time = time.time()
        # timestamp로 join 하기 위하여 시간 포맷을 utc로 통일
        measurement_df['MEASUREMENT_DATE'] = measurement_df.MEASUREMENT_DATETIME.dt.date
        measurement_df['MEASUREMENT_DATE'] = pd.to_datetime(
            measurement_df.MEASUREMENT_DATE, utc=True)

        measurement_df['MEASUREMENT_HOUR'] = measurement_df.MEASUREMENT_DATETIME.dt.hour
        measurement_df['MEASUREMENT_HOURGRP'] = measurement_df.MEASUREMENT_HOUR // self.group_hour

        # 평균값 이용하여 Normalize
        if self.measurement_normalize == MEASUREMENT_NORMALIZATION[0]:
            measurement_df = pd.merge(measurement_df,
                                      self.measurement_mean_df.reset_index().rename(
                                          columns={'VALUE_AS_NUMBER': 'MEAN_VALUE'}),
                                      on='MEASUREMENT_SOURCE_VALUE', how='left')
            measurement_df.VALUE_AS_NUMBER = measurement_df.VALUE_AS_NUMBER / \
                measurement_df.MEAN_VALUE
        # 생체신호 범위를 이용하여 Normalize
        elif self.measurement_normalize == MEASUREMENT_NORMALIZATION[1]:
            measurement_df.VALUE_AS_NUMBER = measurement_df.apply(lambda row:
                                                                  self._clip_measurement(
                                                                      row['MEASUREMENT_SOURCE_VALUE'],
                                                                      row['VALUE_AS_NUMBER']),
                                                                  axis=1)

            # TODO
        group_cols = ['PERSON_ID', 'MEASUREMENT_DATE',
                      'MEASUREMENT_HOURGRP', 'MEASUREMENT_SOURCE_VALUE']

        agg_list = ['count', 'min', 'max', 'mean', 'std']

        measurement_df['VALUE_DIFF'] = measurement_df.groupby(
            group_cols).VALUE_AS_NUMBER.diff()

        measurement_diff_df = pd.pivot_table(measurement_df,
                                             values='VALUE_DIFF', index=group_cols[:-1],
                                             columns='MEASUREMENT_SOURCE_VALUE', aggfunc=['mean', 'max', 'min'])
        measurement_diff_df.columns = pd.MultiIndex.from_tuples(
            [('diff', v) for v in measurement_diff_df.columns])

#     measurement_kurt_df = measurement_df.groupby(group_cols).VALUE_AS_NUMBER.apply(pd.DataFrame.kurt).unstack()
#     measurement_kurt_df.columns = pd.MultiIndex.from_tuples([('kurt', v) for v in measurement_kurt_df.columns])

        measurement_df = measurement_df.groupby(
            group_cols).VALUE_AS_NUMBER.agg(agg_list).unstack()
        measurement_df = pd.concat(
            [measurement_df, measurement_diff_df], axis=1).reset_index().ffill().bfill().fillna(0)

        # 사용한 후 삭제
        del measurement_diff_df
#     del measurement_kurt_df

        # 컬럼 이름 정제 (그룹화 하기 쉽게)
        new_cols = []
        for col in measurement_df.columns:
            if col[1] == "":
                new_cols.append(col[0])

            elif col[0] in agg_list:
                new_cols.append((col[1], col[0]))
            elif col[0] == "diff":
                new_cols.append((col[1][0] + "diff", col[1][1]))
        measurement_df.columns = new_cols

        measurement_df = measurement_df.rename(columns={'MEASUREMENT_DATE': 'DATE',
                                                        'MEASUREMENT_HOURGRP': 'HOURGRP'})

        measurement_col_filename = os.path.join(
            self.task_path, 'measurement_cols.npy')
        if self.is_train:
            # 컬럼 이름 저장
            np.save(measurement_col_filename, np.array(measurement_df.columns))
        else:
            # 컬럼 로드
            measurement_cols = np.load(
                measurement_col_filename, allow_pickle=True)
            new_measurement_list = []
            for col in measurement_cols:
                if col in measurement_df.columns:
                    new_measurement_list.append(measurement_df[col])
                else:
                    new_measurement_list.append(
                        pd.Series([0] * measurement_df.shape[0]))

            measurement_df = pd.concat(new_measurement_list, axis=1)
            measurement_df.columns = measurement_cols
        print("data_loader groupby_hour_measurement time:",
              time.time() - start_time)

        scaler = MinMaxScaler(feature_range=(-1,1))
        measurement_df.iloc[:, 3:] = scaler.fit_transform(measurement_df.iloc[:, 3:])

        if self.autoencoder:
            train_measure, valid_measure = train_test_split(measurement_df.iloc[:, 3:],
                                                                  test_size=self.valid_size)

            self.train_measure = train_measure
            self.valid_measure = valid_measure
        else:
            auto_model = Autoencoder(measurement_df.iloc[:,3:])
            auto_model.load(self.task_path)
            embedded = auto_model.predict(measurement_df.iloc[:,3:])
            measurement_df = pd.concat([measurement_df.iloc[:,:3], pd.DataFrame(embedded)],axis=1)

        return measurement_df

    def make_person_sequence(self):
        start_time = time.time()
        # 환자별로 데이터의 시작시간과 종료시간을 구한다.
        timerange_df = self.cohort_df.groupby('SUBJECT_ID').agg(
            {'COHORT_START_DATE': 'min', 'COHORT_END_DATE': 'max'})
        timerange_df['START_DATE'] = timerange_df.COHORT_START_DATE.dt.date
        timerange_df['START_HOURGRP'] = timerange_df.COHORT_START_DATE.dt.hour // self.group_hour
        timerange_df['END_DATE'] = timerange_df.COHORT_END_DATE.dt.date
        timerange_df['END_HOURGRP'] = timerange_df.COHORT_END_DATE.dt.hour // self.group_hour
        timerange_df = timerange_df.drop(
            ['COHORT_START_DATE', 'COHORT_END_DATE'], axis=1)

        demographic_ary = self.person_df.sort_values(
            'PERSON_ID', ascending=True).values
        condition_ary = self.condition_df.sort_values(
            ['PERSON_ID', 'DATE', 'HOURGRP'], ascending=True).values
        measurement_ary = self.measurement_df.sort_values(
            ['PERSON_ID', 'DATE', 'HOURGRP'], ascending=True).values
        timerange_ary = timerange_df.sort_values(
            'SUBJECT_ID', ascending=True).reset_index().values

        demographic_cols = ["AGE_HOUR", "GENDER"]
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
        print("len of keylist: ", len(key_list),
              "shape of measurement: ", self.measurement_df.shape)

        # 시간대 정보에 따라 데이터를 채워 넣는다
        demographic_idx = condition_idx = measurement_idx = 0
        prev_person_id = None
        prev_conditions = None

        data_cols = list(demographic_cols) + \
            list(measurement_cols) + list(condition_cols)
        data_list = np.zeros((len(key_list), len(data_cols)), dtype=np.float32)

        for idx, row in enumerate(key_list):
            person_id, date, hourgrp = row

            col_start_idx = col_end_idx = 0
            col_end_idx += len(demographic_cols)
            # Demographic 추가
            while True:
                if demographic_idx >= len(demographic_ary):
                    break

                demographic_row = demographic_ary[demographic_idx]
                demographic_person_id = demographic_row[0]
                # 시간 계산을 위해 tz를 동일하게 맞춤.
                demographic_age = datetime.combine(date, datetime_time(hour=hourgrp, tzinfo=timezone.utc)).astimezone(
                    pytz.utc) - demographic_row[1]
                demographic_gender = demographic_row[2]
                demographic_data = [
                    demographic_age.total_seconds() // 3600., demographic_gender]

                state = 0       # 0: 다음 데이터 탐색 1: 맞는 데이터 찾음 2: 맞는 데이터 없음
                if demographic_person_id > person_id:       # 다음 환자로 넘어감
                    state = 2
                elif demographic_person_id == person_id:  # 맞는 데이터
                    state = 1

                if state == 0:                  # 계속 탐색
                    demographic_idx += 1
                elif state == 1:                # 데이터 찾음
                    data_list[idx, col_start_idx:col_end_idx] = demographic_data
                    break
                elif state == 2:                # 맞는 데이터가 없음
                    break

            # Measurement 탐색
            col_start_idx = col_end_idx
            col_end_idx += len(measurement_cols)
            while True:
                if measurement_idx >= len(measurement_ary):
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
                    data_list[idx, col_start_idx:col_end_idx] = measurement_data
                    measurement_idx += 1
                    break
                elif state == 2:                # 맞는 데이터가 없음
                    break

            # Condition 탐색
            col_start_idx = col_end_idx
            col_end_idx += len(condition_cols)
            # 이전과 다른 환자임. condition정보 reset
            if prev_person_id != person_id:
                prev_conditions = np.array([0] * len(condition_cols))

            while True:
                if condition_idx >= len(condition_ary):
                    break

                condition_row = condition_ary[condition_idx]
                condition_person_id = condition_row[0]
                condition_date = condition_row[1]
                condition_hourgrp = condition_row[2]
                condition_data = condition_row[3:]

                state = 0       # 0: 다음 데이터 탐색 1: 맞는 데이터 찾음 2: 맞는 데이터 없음
                if condition_person_id > person_id:       # 다음 환자로 넘어감
                    state = 3
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
                    prev_conditions = np.array(
                        prev_conditions) + np.array(condition_data)
                    data_list[idx, col_start_idx:col_end_idx] = prev_conditions
                    condition_idx += 1
                    break
                elif state == 2:                # 맞는 데이터가 없음
                    data_list[idx, col_start_idx:col_end_idx] = prev_conditions
                    break
                elif state == 3:
                    break

            prev_person_id = person_id

        self.feature_ary = data_list
        self.feature_key_df = pd.DataFrame(
            key_list, columns=['PERSON_ID', 'DATE', 'HOURGRP'])
        print("data_loader make_person_sequence time:", time.time() - start_time)

    def make_data(self):
        start_time = time.time()
        # 빠른 서치를 위하여 데이터 정렬
        # 가장 마지막 시점이 먼저 오도록 반대로 정렬
        cohort_df = self.cohort_df.sort_values(
            ['SUBJECT_ID', 'COHORT_END_DATE'], ascending=[True, False])
        feature_key_df = self.feature_key_df.sort_values(
            ['PERSON_ID', 'DATE', 'HOURGRP'], ascending=[True, False, False])
        feature_ary = self.feature_ary[feature_key_df.index]
        feature_key_ary = feature_key_df.values

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
                person_id, feature_date, feature_hourgrp = feature_key_ary[feature_idx]
                feature_row = feature_ary[feature_idx]

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
                        timestep_person_id = feature_key_ary[feature_idx + timestep][0]
                        timestep_row = feature_ary[feature_idx + timestep]
                        if timestep_person_id == subject_id:
                            timestep_data = timestep_row
                            each_x_list.append(timestep_data)
                        else:
                            break
                    # 가장 나중 데이터부터 each_x_list에 넣었으니 데이터에 넣을땐 반대로
                    x_list.append(np.array(each_x_list)[::-1])
                    break
                elif person_id > subject_id:
                    # 데이터를 못찾음. 다음 환자로 넘어가버렸다
                    print("Person's data not found", subject_id)
                    feature_data = feature_row
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
        self.key = pd.DataFrame(
            key_list, columns=['SUBJECT_ID', 'COHORT_END_DATE'])
        print("X", self.x.shape)
        if self.y is not None:
            print("Y", self.y.shape)
        print("Key", self.key.shape)
        print("data_loader make_data time:", time.time() - start_time)

    def _stratified_shuffle(self):
        whole_patient = set(self.key.SUBJECT_ID.unique())
        true_patient = set(self.key.loc[np.where(
            self.y == 1)[0], ].SUBJECT_ID.unique())
        false_patient = whole_patient - true_patient

        true_train_patient, true_valid_patient = train_test_split(list(true_patient),
                                                                  train_size=(
                                                                      1 - self.valid_size),
                                                                  test_size=self.valid_size,
                                                                  random_state=self.data_split_random_seed)

        false_train_patient, false_valid_patient = train_test_split(list(false_patient),
                                                                    train_size=(
                                                                        1 - self.valid_size),
                                                                    test_size=self.valid_size,
                                                                    random_state=self.data_split_random_seed)

        train_patient = np.concatenate(
            [true_train_patient, false_train_patient])
        valid_patient = np.concatenate(
            [true_valid_patient, false_valid_patient])

        self.train_x = self.x[self.key.SUBJECT_ID.isin(train_patient)]
        self.train_y = self.y[self.key.SUBJECT_ID.isin(train_patient)]

        self.valid_x = self.x[self.key.SUBJECT_ID.isin(valid_patient)]
        self.valid_y = self.y[self.key.SUBJECT_ID.isin(valid_patient)]

    def _train_split_data(self):
        try:
            self._stratified_shuffle()
        except ValueError:  # is sample data
            self.train_x = self.x
            self.train_y = self.y

            self.valid_x = self.x
            self.valid_y = self.y

        self.train_x = pad_sequences(
            self.train_x, dtype=np.float32, padding='post', value = -5)
        self.valid_x = pad_sequences(
            self.valid_x, dtype=np.float32, padding='post', value = -5)

    def split_data(self):
        start_time = time.time()
        if self.is_train:
            self._train_split_data()
        else:
            self.train_x = pad_sequences(self.x)

        print("data_loader split_data time:", time.time() - start_time)

    def get_train_data(self):
        return self.train_x, self.train_y

    def get_valid_data(self):
        return self.valid_x, self.valid_y

    def get_infer_data(self):
        return self.train_x
