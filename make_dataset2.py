import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from tqdm import tqdm
def make_dataset2():
    df2 = pd.read_csv('index_files/df2_mid.csv',parse_dates=['datetime'])
    df2.head(2)

    latlon = np.load('index_files/latlon_grid_hrrr.npy')
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
    points

    eps = 0.2
    min_samples = 2
    def fit_dbscan(points):    
        points = points.to_numpy()
        mins = np.min(points, axis=0)
        maxs = np.max(points, axis=0)
        standardized_ps = (points-mins) / (maxs-mins+1e-6) 
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        #print(standardized_ps.shape)
        dbscan.fit(standardized_ps)
        return dbscan.labels_ 

    dataset2 = []
    for date in tqdm(df2.datetime.unique()):
        part = df2[df2.datetime==date]
        #print(date)
        if len(part)>= 3:
            labels = fit_dbscan(part[['BEGIN_LAT','BEGIN_LON']])
            if not len(np.unique(labels)) == len(labels): # skip if each point is a cluster
                for label in np.unique(labels):
                    if label!=-1:
                        cluster = part.iloc[labels==label]
                        points = find_closest_points(latlon,cluster[['BEGIN_LAT','BEGIN_LON']].values)
                        points = np.stack(points)
                        dataset2.append([date,
                                        '+'.join(cluster.type.unique()),
                                        max(points[:,1].min()-20,0),
                                        max(points[:,0].min()-20,0),
                                        min(points[:,1].max()+20,1798),
                                        min(points[:,0].max()+20,1058)])
    len(dataset2),dataset2[0]   

    result = pd.DataFrame({
        'begin_time':[i[0] for i in dataset2],
        'end_time':[i[0] for i in dataset2],
        'type':[i[1] for i in dataset2],
        'bounding_box':[str(i[2])+'_'+str(i[3])+'_'+str(i[4])+'_'+str(i[5]) for i in dataset2],
    })
    result.to_csv('index_files/extreme_data_info2.csv',index=False)
    print('data info2 finsihed and save!, shape',result.shape)