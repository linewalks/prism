import pandas as pd
import numpy as np
import os

__measurement_file_name = 'sample_measurement_table.csv'
__condition_occurrence_file_name ='sample_condition_occurrence_table.csv'
__outcome_cohort_file_name = 'sample_outcome_cohort_table.csv'
__person_file_name = 'sample_person_table.csv'

def load_data_set(directory):
    person_table = pd.read_csv(os.path.join(directory,__person_file_name), encoding = 'windows-1252')
    outcome_cohort_table = pd.read_csv(os.path.join(directory,__outcome_cohort_file_name), encoding = 'windows-1252')
    condition_occurrence_table = pd.read_csv(os.path.join(directory,__condition_occurrence_file_name), encoding = 'windows-1252')
    measurement_table = pd.read_csv(os.path.join(directory,__measurement_file_name), encoding = 'windows-1252')
    return person_table, condition_occurrence_table, outcome_cohort_table, measurement_table

person_table, condition_table, outcome_table, measure_table = load_data_set('')

actual_list=["HR", "RR", "SpO2", "Pulse", "Temp", "ABPm",
             "ABPd", "ABPs", "NBPm", "NBPs", "NBPd", "SPO2-%",
             "SPO2-R", "Resp", "PVC", "ST-II", "etCO2", "Weight",
             "Height", "SpO2 r", "imCO2", "ST-V1", "ST-AVF", "ST-AVL",
             "ST-AVR", "ST-I", "ST-III", "ST-aVF", "ST-aVL", "ST-aVR",
             "awRR", "CVPm", "AoM", "ST-V2", "ST-V3", "ST-V4", "ST-V5",
             "ST-V6", "SpO2T", "T1", "TV", "Cdyn", "PEEP", "RRaw", "TVin",
             "inO2", "AoD", "AoS", "InsTi", "MINVOL", "MnAwP", "PIP",
             "MVin", "PB", "eeFlow", "Poccl", "Pplat", "MV", "Patm",
             "Pmean", "Ppeak", "Rinsp", "ST-V", "sInsTi", "sPEEP",
             "sTV", "sTrig", "sPSV", "Rexp", "sPltTi", "SpMV", "highP",
             "sAADel", "sAFIO2", "sAPkFl", "sARR", "sATV", "sAWRR",
             "sBasFl", "sFIO2", "sPIF", "sPin", "sSenFl", "sMV", "sO2",
             "sRisTi", "ARTd", "ARTm", "ARTs", "PAPm", "sSIMV", "PAPd",
             "PAPs", "Trect", "sCMV"]

measure_table[['MEASUREMENT_DATE(new)',"MEASUREMENT_DATETIME(new)"]]=\
measure_table[['MEASUREMENT_DATE',"MEASUREMENT_DATETIME"]].apply(lambda x:pd.to_datetime(x))
measure_table['timerenewal']=(measure_table['MEASUREMENT_DATE(new)']+\
                        pd.to_timedelta(measure_table['MEASUREMENT_DATETIME(new)'].dt.hour, unit='h')+pd.to_timedelta(1,unit='h'))

group1=measure_table.groupby(['PERSON_ID','timerenewal'])

code='HR'
mean=measure_table.loc[measure_table["MEASUREMENT_SOURCE_VALUE"]==f'{code}'].groupby(['PERSON_ID','timerenewal'])["VALUE_SOURCE_VALUE"].mean()
std=measure_table.loc[measure_table["MEASUREMENT_SOURCE_VALUE"]==f'{code}'].groupby(['PERSON_ID','timerenewal'])["VALUE_SOURCE_VALUE"].std()
max1=measure_table.loc[measure_table["MEASUREMENT_SOURCE_VALUE"]==f'{code}'].groupby(['PERSON_ID','timerenewal'])["VALUE_SOURCE_VALUE"].max()
min1=measure_table.loc[measure_table["MEASUREMENT_SOURCE_VALUE"]==f'{code}'].groupby(['PERSON_ID','timerenewal'])["VALUE_SOURCE_VALUE"].min()
count=measure_table.loc[measure_table["MEASUREMENT_SOURCE_VALUE"]==f'{code}'].groupby(['PERSON_ID','timerenewal'])["VALUE_SOURCE_VALUE"].count()
diffmean = mean.groupby(['PERSON_ID']).diff()
diffstd = std.groupby(['PERSON_ID']).diff()
diffmax1 = max1.groupby(['PERSON_ID']).diff()
diffmin1 = min1.groupby(['PERSON_ID']).diff()
diffcount = count.groupby(['PERSON_ID']).diff()
diffmean.loc[diffmean.isna()]=0
diffstd.loc[diffstd.isna()]=0
diffmax1.loc[diffmax1.isna()]=0
diffmin1.loc[diffmin1.isna()]=0
diffcount.loc[diffcount.isna()]=0

group=pd.concat([mean, count, std,max1,min1,
                 diffmean, diffcount, diffstd, diffmax1, diffmin1], axis=1)
group.columns=[f"{code}(mean)",f"{code}(count)",f"{code}(std)",f"{code}(max)", f"{code}(min)",
              f"{code}(dmean)",f"{code}(dcount)",f"{code}(dstd)",f"{code}(dmax)",f"{code}(dmin)"]
HR=group

actual_list

for code in actual_list[1:]:
    mean=measure_table.loc[measure_table["MEASUREMENT_SOURCE_VALUE"]==f'{code}'].groupby(['PERSON_ID','timerenewal'])["VALUE_SOURCE_VALUE"].mean()
    std=measure_table.loc[measure_table["MEASUREMENT_SOURCE_VALUE"]==f'{code}'].groupby(['PERSON_ID','timerenewal'])["VALUE_SOURCE_VALUE"].std()
    max1=measure_table.loc[measure_table["MEASUREMENT_SOURCE_VALUE"]==f'{code}'].groupby(['PERSON_ID','timerenewal'])["VALUE_SOURCE_VALUE"].max()
    min1=measure_table.loc[measure_table["MEASUREMENT_SOURCE_VALUE"]==f'{code}'].groupby(['PERSON_ID','timerenewal'])["VALUE_SOURCE_VALUE"].min()
    count=measure_table.loc[measure_table["MEASUREMENT_SOURCE_VALUE"]==f'{code}'].groupby(['PERSON_ID','timerenewal'])["VALUE_SOURCE_VALUE"].count()
    diffmean = mean.groupby(['PERSON_ID']).diff()
    diffstd = std.groupby(['PERSON_ID']).diff()
    diffmax1 = max1.groupby(['PERSON_ID']).diff()
    diffmin1 = min1.groupby(['PERSON_ID']).diff()
    diffcount = count.groupby(['PERSON_ID']).diff()
    diffmean.loc[diffmean.isna()]=0
    diffstd.loc[diffstd.isna()]=0
    diffmax1.loc[diffmax1.isna()]=0
    diffmin1.loc[diffmin1.isna()]=0
    diffcount.loc[diffcount.isna()]=0

    group=pd.concat([mean, count, std,max1,min1,
                     diffmean, diffcount, diffstd, diffmax1, diffmin1], axis=1)
    group.columns=[f"{code}(mean)",f"{code}(count)",f"{code}(std)",f"{code}(max)", f"{code}(min)",
                  f"{code}(dmean)",f"{code}(dcount)",f"{code}(dstd)",f"{code}(dmax)",f"{code}(dmin)"]

    HR=HR.merge(group, on=['PERSON_ID','timerenewal'],how='left')


condition_table[["CONDITION_START_DATE"]].apply(lambda x:pd.to_datetime(x))

condition1=['P07.3', 'P07.1', 'P22.0', 'P22.9', 'P07.2', 'Z38.0', 'Z38.3',
       'P27.1', 'P03.4', 'P07.0', 'Q21.1', 'H35.1', 'R62.0', 'P59.0',
       'P05.1', 'P91.2', 'E03.1', 'Z00.0', 'P01.1', 'D18.0', 'Q25.0',
       'K40.9', 'P00.0', 'Z38.6', 'P28.4', 'R94.6', 'P08.1', 'Z13.9',
       'Q65.2', 'P52.2', 'P01.7', 'H93.2', 'P00.8', 'P70.4', 'B34.9',
       'P29.3', 'P05.9', 'E03.9', 'Z26.9', 'Q04.8', 'H65.9', 'J21.9',
       'P25.1', 'U83.0', 'M43.62', 'D75.8', 'P52.1', 'E27.4', 'P01.0',
       'Q53.2', 'P76.0', 'P02.7', 'P70.0', 'P52.9', 'E80.7', 'P02.0',
       'P01.2', 'Q04.3', 'R56.8', 'P72.2', 'R50.9', 'P00.9', 'Q69.1',
       'Z01.0', 'N43.3', 'R62.8', 'K21.9', 'J06.9', 'E87.1', 'Q53.1',
       'Q42.3', 'P52.8', 'R05', 'N13.3', 'G25.3', 'P04.0', 'Q87.2',
       'L05.9', 'H91.9', 'P92.9', 'D22.9', 'E03.8', 'Q31.5', 'J40',
       'P59.9', 'P29.8', 'J12.3', 'J93.9', 'P56.9', 'P01.5', 'E63.9',
       'N83.2', 'J18.9', 'Q62.0', 'Q82.8', 'P28.2', 'K61.0', 'N48.1',
       'E83.5', 'R34', 'Q63.2', 'K40.3', 'Z38.1', 'Q89.9', 'H66.9',
       'D64.9', 'P96.8', 'P01.3', 'P35.1', 'Q10.5', 'P02.4', 'E22.2',
       'P54.0', 'J21.1', 'Q33.6', 'P26.9', 'P74', 'I28.8', 'N94.8',
       'M67.4', 'R21', 'R25.1', 'L92.8', 'P78.1', 'T81.3', 'P61.2',
       'T18.9', 'I47.1', 'G93.8', 'F98.2', 'P26.1', 'G00.2', 'P02.1',
       'E88.9', 'A49.0', 'L74.3', 'O31.2', 'Q04.6', 'K56.5', 'S09.9',
       'I50.9', 'R09.8', 'E87.2', 'B95.6', 'Q66.8', 'Q64.4', 'P54.3',
       'Q67.3', 'G00.8', 'K92.2', 'Q53.9', 'J98.1', 'P52.6', 'P94.2',
       'K59.0', 'R73.9', 'I27.2', 'R00.1', 'R11', 'P81.9', 'E55.0',
       'N17.9', 'L22', 'Q10.3', 'R68.1', 'P90', 'R04.8', 'R16.2', 'G91.8',
       'P91.7', 'P52.3', 'P61.0', 'P59.8', 'H90.2', 'F19.0', 'Q27.0',
       'D22.5', 'R49.0', 'R09.2', 'R09.3', 'S36.4', 'P52.0', 'K91.4']

for i in range(len(condition1)):
    condition_table.loc[condition_table["CONDITION_SOURCE_VALUE"]==condition1[i],condition1[i]]=1

condition_table['CONDITION_START_DATE']=condition_table['CONDITION_START_DATE'].apply(lambda x:pd.to_datetime(x))

datasetn=pd.concat([condition_table[["PERSON_ID","CONDITION_START_DATE"]],condition_table[condition1]],axis=1)

HR1=HR.reset_index()
total=HR1.merge(datasetn, left_on=["PERSON_ID","timerenewal"],
         right_on=["PERSON_ID","CONDITION_START_DATE"], how='left')

total[condition1]=total.groupby(['PERSON_ID'])[condition1].ffill().iloc[:,1:]

total[condition1]=total[condition1].fillna(0)

total=total.drop(["CONDITION_START_DATE"],axis=1)

total=total.groupby(['PERSON_ID','timerenewal']).ffill()
total=total.groupby(['PERSON_ID','timerenewal']).bfill()
