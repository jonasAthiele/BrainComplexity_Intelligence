# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 10:34:35 2022

@author: Jonas
"""
## Compare associations between MSE and intelligence between main and replication sample

import pandas as pd
import numpy as np
import pickle
import scipy.stats

# load results from main and replication sample
df_main = pd.read_pickle('df_results_main') # associations main sample
df_repli = pd.read_pickle('df_results_repli') # associations replication sample

column_names_main = np.array(df_main.columns)
column_names_repli = np.array(df_repli.columns)
check = (column_names_main[0:-3]==column_names_repli).all() # last three columns of column_names_main are post-hoc analyses done in main sample only
print(check) # check if column names are identical between main and replication sample
column_names = column_names_repli

# load channel names
with open("channel_names", "rb") as fp:
    channel_names = pickle.load(fp)



# compare all associations between intelligence and mse of single scales and channels between main and replication sample
idx_vars = []
for ch in range(28):
    for sc in range(20):
        var_name = 'mse_ch_' + channel_names[ch] + '_sc_' + str(sc)
        idx_vars.append(np.where(column_names==var_name)[0][0])
idx_vars = np.array(idx_vars)

eff_mse_main = np.array(df_main)[0,idx_vars]
eff_mse_repli = np.array(df_repli)[0,idx_vars]
  
corr_effect_mse_RAPM_main_repli = scipy.stats.pearsonr(eff_mse_main,eff_mse_repli)

print('-------------------------------------------')
print('corr patterns mse: corr,p')
print(corr_effect_mse_RAPM_main_repli)
print('-------------------------------------------')

    

# compare all associations between intelligence and clustered mse between main and replication sample
idx_vars = []
n_cluster = 7
n_scale = 4
for ch in range(n_cluster):
    for sc in range(n_scale):
        var_name = 'mse_clustered_cluster_' + str(ch) + '_' + str(sc)    
        idx_vars.append(np.where(column_names==var_name)[0][0])
idx_vars = np.array(idx_vars)

eff_mse_cluster_main = np.array(df_main)[0,idx_vars]
eff_mse_cluster_repli = np.array(df_repli)[0,idx_vars]
    
corr_effect_mseCluster_RAPM_main_repli = scipy.stats.pearsonr(eff_mse_cluster_main, eff_mse_cluster_repli)

print('-------------------------------------------')
print('corr patterns mse clustered: corr,p')
print(corr_effect_mseCluster_RAPM_main_repli )
print('-------------------------------------------')