import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
from tqdm import tqdm
import requests
from io import StringIO
import argparse
'''
first step:
    download the reports from Storm Prediction Center for a given period
    Storm Prediction Center: https://www.spc.noaa.gov/climo/reports/

'''


def make_dataset1(BEGIN_DATETIME,END_DATETIME):
    print(f'start {BEGIN_DATETIME} {END_DATETIME}')

    dates = pd.date_range(start=BEGIN_DATETIME,end=END_DATETIME,freq='D')
    
    dfs = []
    print('start downloading files from Storm Prediction Center')
    for date in tqdm(dates):
        date_str = date.strftime('%Y%m%d')
        for t in ['torn','hail','wind']:
            url = f"https://www.spc.noaa.gov/climo/reports/{date_str[2:]}_rpts_{t}.csv"
            response = requests.get(url)
            response.raise_for_status() 
            csv_content = response.content.decode('utf-8')
            try:
                df = pd.read_csv(StringIO(csv_content))
                if len(df)>0:
                    df['date'] = date_str
                    df['type'] = t
                    dfs.append(df)
            except Exception as e:
                print(url, e)
    dfs = pd.concat(dfs)
    dfs.columns = [i.lower() for i in dfs.columns] # make column names consistent
    dfs['time'] = dfs['time'].apply(lambda x:str(x).zfill(4))
    dfs['datetime'] = dfs['date'] + dfs['time']
    dfs['datetime'] = pd.to_datetime(dfs['datetime'], format='%Y%m%d%H%M')
    dfs['lon'] = dfs['lon'] + 360
    dfs = dfs.reset_index(drop=True)
    
    dfs.to_csv('index_files/df1_mid.csv',index=False)
    dfs = pd.read_csv('index_files/df1_mid.csv',parse_dates=['datetime'])
    ## start doing feature engineering, get ready for later steps
    old = dfs

    old['BEGIN_YEARMONTH'] = old.datetime.apply(lambda x:int(x.strftime("%Y%m")))
    old['BEGIN_DAY'] = old.datetime.apply(lambda x:int(x.strftime("%d")))
    old['location'] = old.location+', ' + old.state + ', ' + old.county
    old['datetime'] = old['datetime'].dt.floor('H')
    old.drop(columns=['date','time','state','county'],inplace=True)

    '''
    read reports from Storm Events Database:
        why load these two together?
        answer: to remove the overlap of data bewteen Storm Prediction Center and
                Storm Event Database
    '''
    new_detail = pd.read_csv('index_files/StormEvents_details-ftp_v1.0_d2020_c20231217.csv')
    df = old[(old['BEGIN_YEARMONTH']>=int(BEGIN_DATETIME[:6]))&(old['BEGIN_YEARMONTH']<=int(END_DATETIME[:6]))]
    new_detail.BEGIN_LON = new_detail.BEGIN_LON + 360
    new_detail.END_LON = new_detail.END_LON + 360


    chosen_cols = ['EVENT_TYPE','BEGIN_YEARMONTH', 'BEGIN_DAY', 'BEGIN_TIME', 'END_YEARMONTH',
        'END_DAY', 'END_TIME', 'EPISODE_ID', 'EVENT_ID', 'STATE', 
        'BEGIN_RANGE', 'END_RANGE', 'BEGIN_LAT', 'END_LAT', 'BEGIN_LON', 'END_LON',
        'EPISODE_NARRATIVE', 'EVENT_NARRATIVE' ]
    new_detail = new_detail[chosen_cols]
    #new_detail['area'] = np.abs(new_detail.BEGIN_LAT-new_detail.END_LAT) * np.abs(new_detail.BEGIN_LON-new_detail.END_LON)*1000
    new_detail.BEGIN_RANGE = new_detail.BEGIN_RANGE*1.61
    new_detail.END_RANGE = new_detail.END_RANGE*1.61
    #new_detail['ABS_DIS'] = np.sqrt((new_detail.BEGIN_LAT-new_detail.END_LAT)**2 +(new_detail.BEGIN_LON-new_detail.END_LON)**2)
    new_detail.isna().mean() * 100

    new_df = new_detail[~new_detail.BEGIN_RANGE.isna()]
    print('data shape from Storm Event Database:',new_df.shape)


    '''
    scripts for convert distance in km to latitude longitude different
    '''
    import math
    # Earth radius in kilometers
    R = 6371.0

    def latitude_difference(distance):
        delta_phi = distance / R
        delta_phi_degrees = math.degrees(delta_phi)
        return delta_phi_degrees

    def longitude_difference(latitude, distance):
        phi = math.radians(latitude)
        delta_lambda = distance / (R * math.cos(phi))
        delta_lambda_degrees = math.degrees(delta_lambda)
        return delta_lambda_degrees

    distance_km = 100  # Example distance in kilometers
    fixed_latitude = 45  # Example fixed latitude in degrees
    fixed_longitude = -75 +360 # Example fixed longitude in degrees

    print(f"Difference in latitude for {distance_km} km at fixed longitude: {latitude_difference(distance_km)} degrees")
    print(f"Difference in longitude for {distance_km} km at latitude {fixed_latitude} degrees: {longitude_difference(fixed_latitude, distance_km)} degrees")


    new_df['BEGIN_YEARMONTH_str'] = new_df['BEGIN_YEARMONTH'].astype(str)
    new_df['BEGIN_DAY_str'] = new_df['BEGIN_DAY'].astype(str).str.zfill(2)
    new_df['END_YEARMONTH_str'] = new_df['END_YEARMONTH'].astype(str)
    new_df['END_DAY_str'] = new_df['END_DAY'].astype(str).str.zfill(2)
    new_df['hour_str'] = new_df['BEGIN_TIME'].astype(str).str.zfill(4).str[:2]
    new_df['end_hour_str'] = new_df['END_TIME'].astype(str).str.zfill(4).str[:2]
    new_df['begin_date'] = new_df['BEGIN_YEARMONTH_str'] + new_df['BEGIN_DAY_str'] + new_df['hour_str']
    new_df['end_date'] = new_df['END_YEARMONTH_str'] + new_df['END_DAY_str'] + new_df['end_hour_str']
    new_df['begin_date'] = pd.to_datetime(new_df['begin_date'], format='%Y%m%d%H')
    new_df['end_date'] = pd.to_datetime(new_df['end_date'], format='%Y%m%d%H')
    new_df.drop(columns = ['BEGIN_YEARMONTH_str', 'BEGIN_DAY_str', 'hour_str','end_hour_str','END_YEARMONTH_str','END_DAY_str'], inplace=True)
    new_df.EVENT_TYPE = new_df.EVENT_TYPE.apply(lambda x:x.replace(' ','_'))
    df.rename(columns={'lat':'BEGIN_LAT','lon':'BEGIN_LON'},inplace=True)
    new_df.head(3)
    print('new df shape',new_df.shape)
    new_df = new_df[(new_df.BEGIN_YEARMONTH>=int(BEGIN_DATETIME[:6]))&(new_df.BEGIN_YEARMONTH<=int(END_DATETIME[:6]))]
    print('new df shape',new_df.shape)

    ## loading the latitude and longitude grid from a zarr file
    latlon = np.load('index_files/latlon_grid_hrrr.npy')
    # the process is shown below
    '''
    data = xr.open_dataset('/pde/hrrr/zarr/20190101.zarr', engine='zarr')
    lat = data.lat.data
    lon = data.lon.data
    latlon = np.stack((lat, lon), axis=-1)
    np.save('index_files/latlon_grid_hrrr.npy',latlon)
    '''
    def find_closest_point(grid, new_point):
        distances = np.sqrt(np.sum((grid - new_point) ** 2, axis=2))
        return np.unravel_index(np.argmin(distances), distances.shape)

    def find_closest_points(grid, new_points):
        # lat lon range (21.138123, 52.615654, 225.90453, 299.0828)
        new_points = np.array(new_points)
        new_points = new_points[:, np.newaxis, np.newaxis, :]
        distances = np.sqrt(np.sum((grid - new_points) ** 2, axis=3))
        reshaped_distances = distances.reshape(distances.shape[0], -1)
        min_flat_indices = np.argmin(reshaped_distances, axis=1)
        min_indices = np.unravel_index(min_flat_indices, distances.shape[1:3])
        closest_indices = list(zip(*min_indices))
        return closest_indices
    points = find_closest_points(latlon,[[35,280],[22,226]])
    points # [(450, 1428), (165, 0)]




    #####
    # start making bounding boxes
    # and store the indexes of which rows are overlapped 
    indexes = []
    print('start making bounding boxes')
    bboxes = []
    for ep in tqdm(new_df.EPISODE_ID.unique()):
        part = new_df[new_df.EPISODE_ID==ep]
        points = []
        begin_dates = []
        end_dates = []
        for idx,row in part.iterrows():
            begin_dates.append(row.begin_date)
            end_dates.append(row.end_date)
            old_part = df[(df.datetime>=row.begin_date)&(df.datetime<=row.end_date)&(df.BEGIN_LAT>row.BEGIN_LAT-1)&
                        (df.BEGIN_LAT<row.BEGIN_LAT+1)&(df.BEGIN_LON>row.BEGIN_LON-1)&(df.BEGIN_LON<row.BEGIN_LON+1)]
            if old_part.shape[0]>0:
                indexes.extend(old_part.index.tolist())
                for _, old_row in old_part.iterrows():
                    points.append([row.BEGIN_LAT,row.BEGIN_LON])
            points.append([row.BEGIN_LAT,row.BEGIN_LON])
            if row.BEGIN_RANGE>0:
                diff_lat = latitude_difference(row.BEGIN_RANGE)
                diff_lon = longitude_difference(row.BEGIN_LAT,row.BEGIN_RANGE)

                points.append([row.BEGIN_LAT+diff_lat,row.BEGIN_LON-diff_lon]) # left_upper
                points.append([row.BEGIN_LAT+diff_lat,row.BEGIN_LON+diff_lon]) # right upper
                points.append([row.BEGIN_LAT-diff_lat,row.BEGIN_LON-diff_lon]) # left_lower 
                points.append([row.BEGIN_LAT-diff_lat,row.BEGIN_LON+diff_lon]) # right_lower

        types = np.unique(part.EVENT_TYPE.unique())
        types_detail = []
        for t in types:
            type_df = part[part.EVENT_TYPE==t]
            types_detail.append(t + ' ' + type_df.begin_date.min().strftime("%Y%m%d%H") + ' ' + type_df.end_date.max().strftime("%Y%m%d%H"))
        types_detail = ','.join(types_detail)
        #print('type detail:',types_detail)
        types_str = '+'.join(types)
        #print('type str:',types_str)
        points = np.array(points)
        points = [[np.min(points[:,0]),np.min(points[:,1])],
                [np.max(points[:,0]),np.max(points[:,1])]]
        points = np.array(find_closest_points(latlon,points))
        points[0] = np.clip(points[0] - 20,0,10000)
        points[1,0] = min(points[1,0] + 20, 1058) 
        points[1,1] = min(points[1,1] + 20, 1798)    # increase the bounding box size by 40
        bboxes.append( [types_str,str(points[0,1])+'_'+str(points[0,0])+'_'+str(points[1,1])+'_'+str(points[1,0]),np.min(begin_dates),np.max(end_dates),types_detail]) 
        # x_min,y_min,x_max,y_max, begin_hour, end_hour

    data1 = pd.DataFrame({
        'type': [i[0] for i in bboxes],
        'bounding_box':[i[1] for i in bboxes],
        'begin_time': [i[2] for i in bboxes],
        'end_time': [i[3] for i in bboxes],
        'details':[i[4] for i in bboxes],
    })
    data1

    data1.to_csv('index_files/extreme_data_info1.csv',index=False)

    df2 =  df[~df.index.isin(indexes)]
    df2 = df2[df2.BEGIN_YEARMONTH>=int(BEGIN_DATETIME[:6])&(df2.BEGIN_YEARMONTH<=int(END_DATETIME[:6]))]
    df2.to_csv('index_files/df2_mid.csv',index=False)

