from make_dataset1 import make_dataset1
from make_dataset2 import make_dataset2
from make_dataset3 import make_dataset3
from datetime import datetime,timedelta
import argparse
import pandas as pd
import os
import numpy as np

parser = argparse.ArgumentParser(description='Process an integer input.')

parser.add_argument('begin_datetime', type=str, help='The begin datetime in format YYYYMMDD')
parser.add_argument('end_datetime', type=str, help='The end datetime in format YYYYMMDD')
# Parse the arguments
args = parser.parse_args()
BEGIN_DATETIME = args.begin_datetime
END_DATETIME = args.end_datetime

make_dataset1(BEGIN_DATETIME,END_DATETIME)
make_dataset2()
make_dataset3(BEGIN_DATETIME,END_DATETIME)

df1 = pd.read_csv('index_files/extreme_data_info1.csv',parse_dates=['begin_time','end_time'])
print(df1.shape)

df2 = pd.read_csv('index_files/extreme_data_info2.csv',parse_dates=['begin_time','end_time'])
print(df2.shape)

df3 = pd.read_csv('index_files/extreme_data_info3.csv',parse_dates=['begin_time','end_time'])
print(df3.shape)

df = pd.concat([df1,df2,df3]).reset_index(drop=True)


######
# adjust the length of span, maximum is 3 days for each event

def adjust_end_date(row):
    delta = min(pd.Timedelta(days=3), row['end_time'] - row['begin_time'])
    return row['begin_time'] + delta

df['end_time'] = df.apply(adjust_end_date, axis=1) # clip the end_time to have at most 3 days for an event
print(df.shape)
df['span'] = df.end_time-df.begin_time + timedelta(hours=4)
#df = df[(df.begin_time>='20200701')&(df.begin_time<'20200702')]
df.span.sum() # Timedelta('1711 days 20:00:00')

def fix_type(string):
    return string.replace('hail','Hail').replace('torn','Tornado').replace('wind','Wind').replace('cold','Cold').replace('heat','Heat')
df['type'] = df['type'].apply(fix_type)

df.to_csv(f'index_files/data_{BEGIN_DATETIME}_{END_DATETIME}_info.csv',index=False)