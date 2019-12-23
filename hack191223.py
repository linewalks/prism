import pandas as pd
import numpy as np
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "last" #"last expr -> all로 바꾸면 전체가 나온다. "
pd.options.display.max_columns = 200
pd.options.display.max_rows = 250
pd.options.display.max_colwidth = 100

dataset=pd.read_csv("./sample_measurement_table.csv",encoding='ms949')


dataset2=pd.read_csv("../생체신호_범위.csv",encoding='ms949')

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

dataset[['MEASUREMENT_DATE(new)',"MEASUREMENT_DATETIME(new)"]]=\
dataset[['MEASUREMENT_DATE',"MEASUREMENT_DATETIME"]].apply(lambda x:pd.to_datetime(x))

dataset["hour"]=\
dataset["MEASUREMENT_DATETIME(new)"].dt.hour

group1=dataset.groupby(['PERSON_ID','MEASUREMENT_DATE','hour'])

dataset["VALUE_SOURCE_VALUE"]
dataset["MEASUREMENT_SOURCE_VALUE"]

actual_list[0]

code='HR'
mean=dataset.loc[dataset["MEASUREMENT_SOURCE_VALUE"]==f'{code}'].groupby(['PERSON_ID','MEASUREMENT_DATE','hour'])["VALUE_SOURCE_VALUE"].mean()
std=dataset.loc[dataset["MEASUREMENT_SOURCE_VALUE"]==f'{code}'].groupby(['PERSON_ID','MEASUREMENT_DATE','hour'])["VALUE_SOURCE_VALUE"].std()
max1=dataset.loc[dataset["MEASUREMENT_SOURCE_VALUE"]==f'{code}'].groupby(['PERSON_ID','MEASUREMENT_DATE','hour'])["VALUE_SOURCE_VALUE"].max()
min1=dataset.loc[dataset["MEASUREMENT_SOURCE_VALUE"]==f'{code}'].groupby(['PERSON_ID','MEASUREMENT_DATE','hour'])["VALUE_SOURCE_VALUE"].min()
count=dataset.loc[dataset["MEASUREMENT_SOURCE_VALUE"]==f'{code}'].groupby(['PERSON_ID','MEASUREMENT_DATE','hour'])["VALUE_SOURCE_VALUE"].count()
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
    mean=dataset.loc[dataset["MEASUREMENT_SOURCE_VALUE"]==f'{code}'].groupby(['PERSON_ID','MEASUREMENT_DATE','hour'])["VALUE_SOURCE_VALUE"].mean()
    std=dataset.loc[dataset["MEASUREMENT_SOURCE_VALUE"]==f'{code}'].groupby(['PERSON_ID','MEASUREMENT_DATE','hour'])["VALUE_SOURCE_VALUE"].std()
    max1=dataset.loc[dataset["MEASUREMENT_SOURCE_VALUE"]==f'{code}'].groupby(['PERSON_ID','MEASUREMENT_DATE','hour'])["VALUE_SOURCE_VALUE"].max()
    min1=dataset.loc[dataset["MEASUREMENT_SOURCE_VALUE"]==f'{code}'].groupby(['PERSON_ID','MEASUREMENT_DATE','hour'])["VALUE_SOURCE_VALUE"].min()
    count=dataset.loc[dataset["MEASUREMENT_SOURCE_VALUE"]==f'{code}'].groupby(['PERSON_ID','MEASUREMENT_DATE','hour'])["VALUE_SOURCE_VALUE"].count()
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

    HR=HR.merge(group, on=['PERSON_ID','MEASUREMENT_DATE','hour'],how='left')


dataset3=pd.read_csv("./sample_condition_occurrence_table.csv",encoding='ms949')
dataset3[["CONDITION_START_DATE"]].apply(lambda x:pd.to_datetime(x))
dataset4=pd.read_csv("../진단코드_목록.csv", encoding='ms949')

condition1=dataset4['CONDITION_SOURCE_VALUE'].unique()
for i in range(len(condition1)):
    dataset3.loc[dataset3["CONDITION_SOURCE_VALUE"]==condition1[i],condition1[i]]=1

dataset3['CONDITION_START_DATE']=dataset3['CONDITION_START_DATE'].apply(lambda x:pd.to_datetime(x))

datasetn=pd.concat([dataset3[["PERSON_ID","CONDITION_START_DATE"]],dataset3[condition1]],axis=1)

HR=HR.reset_index()
HR["MEASUREMENT_DATE"]=HR["MEASUREMENT_DATE"].apply(lambda x:pd.to_datetime(x))
total=HR.merge(datasetn, left_on=["PERSON_ID","MEASUREMENT_DATE"],
         right_on=["PERSON_ID","CONDITION_START_DATE"], how='left')

total[condition1]=total.groupby(['PERSON_ID'])[condition1].ffill().iloc[:,1:]

total[condition1]=total[condition1].fillna(0)
