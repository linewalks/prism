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

measurement_cols = ['PERSON_ID', 'MEASUREMENT_DATETIME', 'MEASUREMENT_SOURCE_VALUE', 'VALUE_AS_NUMBER']
condition_cols = ['PERSON_ID', 'CONDITION_START_DATETIME', 'CONDITION_SOURCE_VALUE']

def generate_measurement(n=10):
  data = []
  for k, stats in MEASUREMENT_SOURCE_VALUE_STATS.items():
    values = np.random.normal(loc=stats['AVERAGE'], scale=stats['STANDARD DEVIATION'], size=n)
    
    data.extend([(k, v) for v in values])
  return data

def generate_condition(n=5):
  return random.choices(condition_list, k=n)

def randomtimes(stime='2019-01-01 00:00:00', etime='2020-03-01 00:00:00', n=10):
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


class Test_DataGenerator():
  def test_generate_measurement(self):
    n = 10
    generated_data = generate_measurement(n)
    assert len(generated_data) == n * len(MEASUREMENT_SOURCE_VALUE_STATS)

  def test_generate_condition(self):
    n = 5
    generated_data = generate_condition(n)
    assert len(generated_data) == n

  def test_generate_time(self):
    n = 10
    generated_data = randomtimes(n=n)
    assert len(generated_data) == n
    
  def test_generate_infants(self):
    n = 3
    n_cond = 5
    n_msmt = 10
    cond_list = []
    msmt_list = []
    for i in range(n):
      for dt, cond in zip(randomtimes(n=n_cond), generate_condition(n_cond)):
        cond_list.append((i, dt, cond))
        for dt2, msmt in zip(randomtimes(stime=dt, etime=None, n=n_msmt), generate_measurement(n_msmt)):
          msmt_list.append((i, dt2, msmt[0], msmt[1]))

    condition_df = pd.DataFrame(cond_list, columns=condition_cols)
    measurement_df = pd.DataFrame(msmt_list, columns=measurement_cols)
    COHORT_START_DATE = min(condition_df.CONDITION_START_DATETIME.min(), 
                            measurement_df.MEASUREMENT_DATETIME.min())
    COHORT_END_DATE = max(condition_df.CONDITION_START_DATETIME.max(), 
                          measurement_df.MEASUREMENT_DATETIME.max())
    print(COHORT_START_DATE, COHORT_END_DATE)

    
