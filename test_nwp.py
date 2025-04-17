import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
import argparse
import os
parser = argparse.ArgumentParser(description='Process two date range')
parser.add_argument('start_date', type=str, help='The first string',default='20200701')
parser.add_argument('end_date', type=str, help='The second string',default='20210101')
args = parser.parse_args()
single_vars = ["msl", "2t", "10u", "10v"]
atmos_vars = ["hgtn", "u", "v", "t", "q"]
atmos_levels = [5000., 10000., 15000., 20000., 
                25000., 30000., 40000., 50000., 
                60000., 70000., 85000., 92500., 
                100000.]

var_name = single_vars + [f"{v}_{int(p/100)}" for v in atmos_vars for p in atmos_levels]
start_date = pd.to_datetime(args.start_date,format='%Y%m%d')
end_date = pd.to_datetime(args.end_date,format='%Y%m%d')
print(f'cal loss for date {start_date} <= date <{end_date}')
ALL_MEANS = np.array([1.0154e+05,  2.8764e+02,  3.6334e-01, -2.1942e-01,  2.0653e+04,
    1.6401e+04,  1.3901e+04,  1.2086e+04,  1.0639e+04,  9.4158e+03,
    7.3890e+03,  5.7308e+03,  4.3219e+03,  3.0932e+03,  1.4998e+03,
    7.9164e+02,  1.2938e+02,  1.9525e+00,  1.3733e+01,  2.1062e+01,
    2.3357e+01,  2.1682e+01,  1.9152e+01,  1.4822e+01,  1.1510e+01,
    8.7861e+00,  6.3429e+00,  2.7486e+00,  1.2042e+00,  3.4059e-01,
    -1.9104e-01,  1.5638e-01,  3.5123e-01,  4.5588e-01,  4.4878e-01,
    3.9698e-01,  2.3847e-01,  1.6056e-01,  1.3170e-01,  1.8216e-01,
    3.9488e-01,  3.1365e-01, -2.5557e-01,  2.1192e+02,  2.0901e+02,
    2.1325e+02,  2.1837e+02,  2.2547e+02,  2.3356e+02,  2.4815e+02,
    2.5946e+02,  2.6831e+02,  2.7558e+02,  2.8359e+02,  2.8666e+02,
    2.9031e+02,  2.8537e-06,  3.0846e-06,  6.6055e-06,  2.4169e-05,
    7.3131e-05,  1.6582e-04,  4.9159e-04,  1.0437e-03,  1.8908e-03,
    2.9435e-03,  5.3577e-03,  7.2076e-03,  8.8479e-03])
ALL_STDS = np.array([6.8789e+02, 1.1408e+01, 3.4087e+00, 3.8128e+00, 1.9669e+02, 2.2751e+02,
    2.8561e+02, 3.0650e+02, 2.9603e+02, 2.7141e+02, 2.1778e+02, 1.7208e+02,
    1.3423e+02, 1.0113e+02, 6.2827e+01, 5.5131e+01, 5.6151e+01, 9.4211e+00,
    1.0859e+01, 1.3950e+01, 1.7115e+01, 1.7562e+01, 1.6353e+01, 1.3276e+01,
    1.0801e+01, 8.9426e+00, 7.5022e+00, 6.1020e+00, 5.4538e+00, 3.7264e+00,
    4.3335e+00, 7.7346e+00, 1.2055e+01, 1.6379e+01, 1.7496e+01, 1.6438e+01,
    1.3244e+01, 1.0723e+01, 8.8519e+00, 7.4927e+00, 6.6357e+00, 6.3794e+00,
    4.4868e+00, 4.5961e+00, 6.9949e+00, 5.8210e+00, 4.5001e+00, 4.9734e+00,
    6.2033e+00, 7.2265e+00, 7.4370e+00, 7.7510e+00, 8.4697e+00, 9.9353e+00,
    1.0570e+01, 1.0831e+01, 1.0117e-06, 2.1395e-06, 4.7380e-06, 1.8255e-05,
    6.3045e-05, 1.5294e-04, 4.8075e-04, 9.9271e-04, 1.6142e-03, 2.2498e-03,
    3.5426e-03, 4.3892e-03, 5.0458e-03])
def cal_nwp_loss(nwp,gt):
    gt_file = gt['targets'][:,0]
    nwp_file = nwp['targets']
    
    mask = torch.tensor(gt['masks'])
    #gt_file = gt_file* ALL_STDS.reshape(1,1,69,1,1) + ALL_MEANS.reshape(1,1,69,1,1)
    nwp_file = torch.tensor(nwp_file)
    gt_file = torch.tensor(gt_file)
    
    loss = masked_mse_loss(nwp_file,gt_file,mask)
    return loss
def masked_mse_loss(pred, target, mask):
    # Compute squared error
    squared_error = (pred - target) ** 2
    squared_error = (squared_error[:,0]*mask[:,None,:,:]).sum(dim=(0,2,3))  / mask.sum() # 1,1,69,320,320  
    #squared_error = squared_error.mean(dim=(0,1,3,4)) #1,1,69,320,320  
    loss = squared_error.sqrt()#* all_stds
    return loss.unsqueeze(0)

losses = []
nwp_root = '/home/v-nianran/kmsw/kmsw0eastus/nian/extreme_dataset/202007_nwp'
gt_root = '/home/v-nianran/kmsw/kmsw0eastus/nian/extreme_dataset/202007'
save_root = '/home/v-nianran/kmsw/kmsw0eastus/nian/extreme_dataset/202007_nwp_test'
gt_files = os.listdir(gt_root)

new_gt_files = []
for f in gt_files:
    date = pd.to_datetime(f.split('_')[0],format='%Y%m%d%H')
    if date<end_date and date>=start_date:
        new_gt_files.append(f)

        
for f in tqdm(new_gt_files):
    #'2020070100_nwp_Flash_Flood+Flood_830_860_929_910.npz
    date = pd.to_datetime(f.split('_')[0],format='%Y%m%d%H')
    nwp_name = f[:10] + '_nwp' + f[10:]
    if date<end_date and date>=start_date:
        nwp = np.load(os.path.join(nwp_root,nwp_name))
        gt = np.load(os.path.join(gt_root,f))
        loss = cal_nwp_loss(nwp,gt)
        losses.append(loss)
losses = torch.cat(losses,dim=0)
#mean_loss = losses.mean(dim=0)
#result = pd.DataFrame({'var':var_name,'nwp_loss':mean_loss.tolist()})
start_date_str = start_date.strftime("%Y%m%d")
end_date_str = end_date.strftime("%Y%m%d")
np.save(os.path.join(save_root,f'nwp_loss_{start_date_str}_{end_date_str}.npy'),losses.numpy())
#result.to_csv(os.path.join(save_root,f'nwp_loss_{start_date_str}.csv'))