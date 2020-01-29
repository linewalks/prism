import os
import numpy as np
import pandas as pd
import random
import datetime
from measurement_stat import MEASUREMENT_SOURCE_VALUE_STATS

condition_list = [
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

label_ratio = 0.1

person_cols = ['PERSON_ID', 'BIRTH_DATETIME', 'GENDER_SOURCE_VALUE']
condition_cols = ['PERSON_ID', 'CONDITION_START_DATETIME', 'CONDITION_SOURCE_VALUE']
measurement_cols = ['PERSON_ID', 'MEASUREMENT_DATETIME', 'MEASUREMENT_SOURCE_VALUE', 'VALUE_AS_NUMBER']
outcome_cols = ['SUBJECT_ID', 'COHORT_START_DATE', 'COHORT_END_DATE', 'LABEL']


def generate_person(n=3):
  data = []
  gender = ['M', 'F']
  for i in range(n):
    data.append((i, randomtimes(n=1)[0], gender[i % 2]))
  return data

def generate_condition(n=5):
  return random.choices(condition_list, k=n)

def generate_measurement(n=5):
  data = []
  for k, stats in MEASUREMENT_SOURCE_VALUE_STATS.items():
    values = np.random.normal(loc=stats['AVERAGE'], scale=stats['STANDARD DEVIATION'], size=n)
    
    data.extend([(k, v) for v in values])
  np.random.shuffle(data)
  return data

def generate_outcome_dts(stime, etime):
  frmt = '%Y-%m-%d %H:%M:%S'
  stime = datetime.datetime.strptime(stime, frmt).replace(minute=00, second=00) + datetime.timedelta(hours=1)
  etime = datetime.datetime.strptime(etime, frmt) + datetime.timedelta(hours=1)
  data = pd.date_range(start=stime, end=etime, freq='H')
   
  return data


def randomtimes(stime='2019-01-01 00:00:00', etime='2019-02-01 00:00:00', n=10):
  frmt = '%Y-%m-%d %H:%M:%S'
  stime = datetime.datetime.strptime(stime, frmt)
  if etime is None:
    etime = stime + datetime.timedelta(days=2)
  else:
    etime = datetime.datetime.strptime(etime, frmt)
  td = etime - stime
  dts = [(random.random() * td + stime).strftime(frmt) for _ in range(n)]
  dts.sort()
  return dts


def generate_dev_data(n=3, n_cond= 5, n_msmt=5):
  
  cond_list = []
  msmt_list = []
  person_list = generate_person(n)
  outcome_list = []
  for person in person_list:
    # 환자를 생성
    i = person[0]
    min_date = person[1]
    max_date = person[1]
    # 진단을 생성
    for dt, cond in zip(randomtimes(stime=min_date, n=n_cond), generate_condition(n_cond)):
      cond_list.append((i, dt, cond))
      if dt > max_date: # 가장 나중날짜를 코호트 종료날짜로
        max_date = dt
      # 바이탈사인 날짜 생성. 측정날짜는 진단 날짜부터
      msmt_dts = randomtimes(stime=dt, n=n_msmt * len(MEASUREMENT_SOURCE_VALUE_STATS))
      for dt2, msmt in zip(msmt_dts, generate_measurement(n_msmt)):
        msmt_list.append((i, dt2, msmt[0], msmt[1]))
        if dt2 > max_date:
          max_date = dt2 # 가장 나중날짜를 코호트 종료날짜로
    
    # 코호트 종료날짜 1시간단위로 롤링
    end_dts = generate_outcome_dts(min_date, max_date)
    start_dts = [min_date] * len(end_dts)
    subject_ids = [i] * len(end_dts)
    # 언발란스로 라벨 생성
    labels = np.random.choice([0, 1], size=len(end_dts), p=[1-label_ratio, label_ratio])
    outcome_list.extend(list(zip(subject_ids, start_dts, end_dts, labels)))

  # 샘플 데이터프레임 생성
  person_df = pd.DataFrame(person_list, columns=person_cols)
  condition_df = pd.DataFrame(cond_list, columns=condition_cols).sort_values(condition_cols[:2])
  measurement_df = pd.DataFrame(msmt_list, columns=measurement_cols).sort_values(measurement_cols[:2])
  outcome_df = pd.DataFrame(outcome_list, columns=outcome_cols)

  # 샘플 데이터프레임 저장
  data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..', 'data')
  dev_data = 'dev'
  csv_files = {
    'person': f'{dev_data}PERSON_NICU.csv',
    'condition': f'{dev_data}CONDITION_OCCURRENCE_NICU.csv',
    'measurement': f'{dev_data}MEASUREMENT_NICU.csv',
    'outcome': f'{dev_data}OUTCOME_COHORT.csv'
  }
  person_df.to_csv(os.path.join(data_path, 'train', csv_files['person']), index=False)
  condition_df.to_csv(os.path.join(data_path, 'train', csv_files['condition']), index=False)
  measurement_df.to_csv(os.path.join(data_path, 'train', csv_files['measurement']), index=False)
  outcome_df.to_csv(os.path.join(data_path, 'train', csv_files['outcome']), index=False)

  print()
  print('generated data shapes')
  print(person_df.shape)
  print(condition_df.shape)
  print(measurement_df.shape)
  print(outcome_df.shape)
