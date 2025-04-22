import pandas as pd
from tqdm import tqdm 
import numpy as np
import os
from sklearn.cluster import DBSCAN
heat_threshold = 37
cold_threshold = -29
hrrr_root = "/kms1_westus3/hrrr/hourly2_fixed_TMP_L103"
def get_extreme_cases(start_date,end_date,season='summer'):
    year = start_date[:4]
    if season=='summer':
        date_range = pd.date_range(start=year+'0601', end=year+'0901',freq='h')
        date_range = date_range[(date_range.hour >= 12) & (date_range.hour <= 20)]
    else:
        date_range = pd.date_range(start=year+'1201', end=year+'1231',freq='h')
        date_range1 = pd.date_range(start=year+'0101', end=year+'0228',freq='h')
        date_range = date_range.append(date_range1)
    start_date = pd.to_datetime(start_date,format='%Y%m%d')
    end_date = pd.to_datetime(end_date,format='%Y%m%d')
    date_range = date_range[(date_range >= start_date) & (date_range <= end_date)]
    dates = []
    point_numbers = []
    for single_date in tqdm(date_range):
        date_str = single_date.strftime("%Y%m%d%H")
        file_path = os.path.join(hrrr_root, f'{date_str}.npy')
        if os.path.exists(file_path):
            try:
                data = np.load(file_path)
                img = data[0,1] # '2t' variable
                img = img - 273.15  # K to C
                img = img.reshape(-1)
                if season=='summer':
                    counts = (img>heat_threshold).sum()
                elif season=='winter':
                    counts = (img<cold_threshold).sum()
                if counts>100:
                    dates.append(single_date)
                    point_numbers.append(counts)
            except:
                print(f'{date_str} file open error')
        else:
            print(f"File does not exist for date: {date_str}")
    return dates,point_numbers


import argparse
parser = argparse.ArgumentParser(description='Process an integer input.')

parser.add_argument('begin_datetime', type=str, help='The begin datetime in format YYYYMMDD')
parser.add_argument('end_datetime', type=str, help='The end datetime in format YYYYMMDD')
# Parse the arguments
args = parser.parse_args()



def make_dataset3(start_date,end_date):
    print(f'start {start_date} {end_date}, dataset3')
    heat_dates,heat_point_numbers = get_extreme_cases(start_date,end_date,season='summer')
    cold_dates,cold_point_numbers = get_extreme_cases(start_date,end_date,season='winter')

    #  you can store for easier loading
    '''
    import pickle
    with open('index_files/cold_dates.pkl', 'wb') as file:
        pickle.dump(cold_dates, file)
    with open('index_files/heat_dates.pkl', 'wb') as file:
        pickle.dump(heat_dates, file)
    '''
    print(f'heat dates : {len(heat_dates)} cold dates: {len(cold_dates)}')

    eps = 0.1
    min_samples = 2
    def fit_dbscan(points):    
        if isinstance(points,pd.DataFrame):
            points = points.to_numpy()
        mins = np.min(points, axis=0)
        maxs = np.max(points, axis=0)
        standardized_ps = (points-mins) / (maxs-mins+1e-6) 
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        #print(standardized_ps.shape)
        dbscan.fit(standardized_ps)
        return dbscan.labels_ 

    dataset3 = []
    for date in tqdm(heat_dates):
        date_str = date.strftime("%Y%m%d%H")
        file_path = os.path.join(hrrr_root, f'{date_str}.npy')
        data = np.load(file_path)
        img = data[0,1]
        img = img - 273.15  
        y, x = np.where(img > 38)    ######### careful   latitude, longitude
        #print(date)
        if len(y)>= 100:
            points = np.concatenate([y[:,None],x[:,None]],axis=1)   
            labels = fit_dbscan(points)
            if not len(np.unique(labels)) == len(labels): # skip if each point is a cluster
                for label in np.unique(labels):
                    if label!=-1:
                        cluster = points[labels==label]
                        dataset3.append([date,'heat',max(cluster[:,1].min()-10,0),
                                        max(cluster[:,0].min()-10,0),
                                        min(cluster[:,1].max()+10,1798),
                                        min(cluster[:,0].max()+10,1058)])
    len(dataset3),dataset3[0]  
    for date in tqdm(cold_dates):
        date_str = date.strftime("%Y%m%d%H")
        file_path = os.path.join(hrrr_root, f'{date_str}.npy')
        data = np.load(file_path)
        img = data[0,1]
        img = img - 273.15  
        y, x = np.where(img <-29)    ######### careful
        #print(date)
        if len(y)>= 3:
            points = np.concatenate([y[:,None],x[:,None]],axis=1)
            labels = fit_dbscan(points)
            if not len(np.unique(labels)) == len(labels): # skip if each point is a cluster
                for label in np.unique(labels):
                    if label!=-1:
                        cluster = points[labels==label]
                        dataset3.append([date,'cold',max(cluster[:,1].min()-10,0),
                                        max(cluster[:,0].min()-10,0),
                                        min(cluster[:,1].max()+10,1798),
                                        min(cluster[:,0].max()+10,1058)])
    print(len(dataset3),dataset3[0],'in dataset3')  
    result = pd.DataFrame({
        'begin_time': [i[0] for i in dataset3],
        'end_time': [i[0] for i in dataset3],
        'type': [i[1] for i in dataset3],
        'bounding_box': [str(i[2])+'_'+str(i[3]) +'_'+str(i[4]) +'_'+str(i[5]) for i in dataset3],
    })
    result.to_csv('index_files/extreme_data_info3.csv',index=False)
 

