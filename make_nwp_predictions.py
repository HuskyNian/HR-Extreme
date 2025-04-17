from herbie import FastHerbie
from concurrent.futures import ThreadPoolExecutor,as_completed
import numpy as np
import pandas as pd
from make_dataset import get_cropped_images
import matplotlib.pyplot as plt
import os
from datetime import datetime,timedelta
import json
from tqdm import tqdm
import warnings
import argparse
single_vars = ["msl", "2t", "10u", "10v"]
atmos_vars = ["hgtn", "u", "v", "t", "q"]
atmos_levels = [5000., 10000., 15000., 20000., 
                25000., 30000., 40000., 50000., 
                60000., 70000., 85000., 92500., 
                100000.]

var_name = single_vars + [f"{v}_{int(p/100)}" for v in atmos_vars for p in atmos_levels]
var_name = np.array(var_name)
herbie_maps = {'hgtn': 'HGT',
    'u': 'UGRD',
    'v': 'VGRD',
    't': 'TMP',
    'q': 'SPFH',
    'msl': 'MSLMA:mean sea level',
    '2t': 'TMP:2 m above',
    '10u': 'UGRD:10 m above',
    '10v': 'VGRD:10 m above'}
new_var_names = []
for var in var_name:
    if var in ['msl','2t','10u','10v']:
        new_var_names.append(herbie_maps[var])
    else:
        var,level_ = var.split('_')
        searchString = f"{herbie_maps[var]}:{level_} mb"
        new_var_names.append(searchString)
            
def load_variable(H_pre,searchString):
    return H_pre.xarray(searchString).to_array().values[0]

def multi_core(H_pre,var_name,max_workers=16):
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_variable = {executor.submit(load_variable, H_pre, var): var for var in var_name}
        for future in as_completed(future_to_variable):
            variable = future_to_variable[future]
            try:
                data = future.result()
                results[variable] = data
            except Exception as exc:
                print(f'{variable} generated an exception: {exc}')
    return results


'''
def load_files_herbie(filenames,new_var_names=new_var_names):
    #### multi core version
    filenames = [f.strftime("%Y%m%d%H")+'.npy' for f in filenames]
    files = {}
    #filenames = ['2019051108.npy', '2019051109.npy', '2019051110.npy']
    if len(filenames)==0:
        return files
    for f in filenames:
        date_str = f.split('.')[0]
        files[date_str] = np.zeros([1,69,1059,1799])
        date = pd.to_datetime(date_str, format='%Y%m%d%H')
        H_pre = FastHerbie([date], model="hrrr", fxx=[1],product='prs',max_threads=50)
        pre_array = multi_core(H_pre,new_var_names)
        for i,var in enumerate(new_var_names):
            files[date_str][0,i] = pre_array[var]
    return files
'''

def load_files_herbie(filenames,new_var_names=new_var_names):
    #### single-core version
    filenames = [f.strftime("%Y%m%d%H")+'.npy' for f in filenames]
    files = {}
    #filenames = ['2019051108.npy', '2019051109.npy', '2019051110.npy']
    if len(filenames)==0:
        return files
    for f in filenames:
        date_str = f.split('.')[0]
        files[date_str] = np.zeros([1,69,1059,1799])
        date = pd.to_datetime(date_str, format='%Y%m%d%H')
        H_pre = FastHerbie([date], model="hrrr", fxx=[1],product='prs',max_threads=50)
        #pre_array = multi_core(H_pre,new_var_names)
        for i,var in enumerate(new_var_names):
            files[date_str][0,i] = H_pre.xarray(var).to_array().values[0]
    return files


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Process an integer input.')

    parser.add_argument('begin_datetime', type=str, help='The begin datetime in format YYYYMMDD')
    parser.add_argument('end_datetime', type=str, help='The end datetime in format YYYYMMDD')
    # Parse the arguments
    args = parser.parse_args()
    df = pd.read_csv('/kmsw/extreme_dataset/data_all202007_info.csv',parse_dates=['begin_time','end_time'])
    #df = pd.read_csv('/home/v-nianran/kmsw/kmsw0eastus/nian/extreme_dataset/data_all202007_info.csv',parse_dates=['begin_time','end_time'])
    save_dir = "/kmsw/extreme_dataset/202007_nwp"
    #save_dir = "/home/v-nianran/kmsw/kmsw0eastus/nian/extreme_dataset/202007_nwp"
    df = df[(df.begin_time>=args.begin_datetime)&(df.begin_time<args.end_datetime)]
    #print('len df',len(df))
    
    after = 2 # output images number
    print('single core version')
    for idx, row in tqdm(df.iterrows(),total=len(df)):
        datetimes_all = pd.date_range(start=row['begin_time'],end=row['end_time'], freq='h')
        x_min,y_min, x_max,y_max = [int(i) for i in row['bounding_box'].split('_')]
        gap = 12 # each time process 24 hours at most
        for i in range(0,len(datetimes_all),gap):
            datetimes = datetimes_all[i:i+gap]      
            need_save = False
            for timestamp in datetimes: # timestamp ['2020-07-01 16:00:00', '2020-07-01 17:00:00','2020-07-01 18:00:00', '2020-07-01 19:00:00','2020-07-01 20:00:00']
                save_name =  '_'.join([timestamp.strftime("%Y%m%d%H"),'nwp',row['type'],row['bounding_box']]) + '.npz'
                if not os.path.exists(os.path.join(save_dir,save_name)):
                    need_save = True

            if need_save:
                imgs = load_files_herbie(datetimes,new_var_names=new_var_names)
                # imgs = dict {'2020070116':np.array(1,69,1059,1799)}
                for timestamp in datetimes: # timestamp ['2020-07-01 16:00:00', '2020-07-01 17:00:00','2020-07-01 18:00:00', '2020-07-01 19:00:00','2020-07-01 20:00:00']
                    save_name =  '_'.join([timestamp.strftime("%Y%m%d%H"),'nwp',row['type'],row['bounding_box']]) + '.npz'
                    if not os.path.exists(os.path.join(save_dir,save_name)):
                        cropped_images,masks = get_cropped_images(imgs,[timestamp],x_min, x_max, y_min, y_max)
                        cropped_images = np.stack(cropped_images)
                        # (4,5,69,320,320) numberOfImagesToCoverArea, timestamp, channels, height, width  
                        masks = np.concatenate(masks,axis=0)
                        #print('save name:',save_name)
                        np.savez(os.path.join(save_dir,save_name),targets=cropped_images,masks=masks)
                    else:
                        #print(save_name,'exist')
                        1 == 1


    #cp make_dataset_final.py /home/msrai4srl4s/nian/blob/home/v-nianran/kmsw/kmsw0eastus0eastau/data/hrrr/nian_extreme_code/make_dataset_final.py