# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 15:04:27 2022

@author: Jonas A. Thiele
"""

#%% Imports

import pandas as pd
import numpy as np
import pingouin as pg
from matplotlib import pyplot as plt
from sklearn import linear_model
import scipy.stats
from scipy.ndimage import measurements
from sklearn.preprocessing import minmax_scale
from statsmodels.stats.multitest import fdrcorrection
import seaborn as sns
import pickle


#%% Load data
beh_repli = pd.read_csv('beh_data_repli.csv', delimiter=';', encoding= 'unicode_escape')
beh_repli = beh_repli[beh_repli['Age'].notna()]
beh_repli = beh_repli[beh_repli['Sex'].notna()]
beh_repli = beh_repli[beh_repli['APM'].notna()]

intell_repli = beh_repli['APM'].to_numpy() # RAPM scores
age_repli = beh_repli['Age'].to_numpy() # age
sex_repli = beh_repli['Sex'].to_numpy() # sex
subs_beh = beh_repli['VPCODE'].to_numpy() # subject IDs corresponding to behavioral data


df_complexity = pd.read_pickle('./df_complexity_repli.pkl') # load complexity measures 
subs_neuro = np.array(df_complexity.IDs_subjects_neuro) # subject IDs corresponding to complexity data
epochs_rejected = np.array(df_complexity.epochs_rejected) # epochs removed during preprocessing

# channel names
with open("channel_names", "rb") as fp:
    channel_names = pickle.load(fp)

#%% Sort behavioral data according to order of subjects in neuro data

idx_neuro = []
no_data_idx_neuro = []
for s in list(subs_neuro):

    idx = np.where(subs_beh == s)[0]
    if idx.shape[0] == 0:
        no_data_idx_neuro.append(np.where(subs_neuro==s))
    else:
        idx_neuro.append(np.where(subs_beh == s)[0])

# remove neuro data of subject without behavioral data        
no_data_idx_neuro = np.array(no_data_idx_neuro).ravel()  
df_complexity.drop(no_data_idx_neuro, axis = 0, inplace=True)  
df_complexity.reset_index(drop=True, inplace = True)
epochs_rejected = np.delete(epochs_rejected, no_data_idx_neuro)
    
age_sorted = age_repli[np.concatenate(idx_neuro)]
sex_sorted = sex_repli[np.concatenate(idx_neuro)]
intel_sorted = intell_repli[np.concatenate(idx_neuro)]

# visualize distribution of RAPM scores
plt.figure()
ax=sns.histplot(intel_sorted,bins=20, color = 'gray', alpha = 0.5)
plt.ylabel('frequency', fontsize = '16') 
plt.xlabel('RAPM', fontsize = '16') 
plt.yticks(fontsize = '16')
plt.xticks(fontsize = '16')
plt.xlim(11,39)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig('intel_repli.jpg', format='jpg', dpi = 1200, bbox_inches='tight')

# visualize distribution of ages
plt.figure()
ax=sns.histplot(age_sorted,bins=20, color = 'gray', alpha = 0.5, )
plt.ylabel('frequency', fontsize = '16') 
plt.xlabel('age', fontsize = '16') 
plt.yticks(fontsize = '16')
plt.xticks(fontsize = '16')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig('age_repli.jpg', format='jpg', dpi = 1200, bbox_inches='tight')

# visualize distribution of rejected epochs
plt.figure()
ax=sns.histplot(epochs_rejected, color = 'gray', alpha = 0.5)
plt.ylabel('frequency', fontsize = '16') 
plt.xlabel('rejected epochs', fontsize = '16') 
plt.yticks(fontsize = '16')
plt.xticks(fontsize = '16')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig('epochs_rejected_repli.jpg', format='jpg', dpi = 1200, bbox_inches='tight')

  
#%% Compute associations between complexity values and RAPM scores 

def get_association(X, y, C):
    
    C = np.array(C)
    X = minmax_scale(X, feature_range=(0, 1))
    data = np.vstack((X, y, C)).T
    
    if C.shape[0] == 2:
        df_data = pd.DataFrame(data, columns=['X','y','conf1','conf2'])
        res = pg.partial_corr(data=df_data, x='X', y='y', covar=['conf1','conf2'])
    
    elif C.shape[0] == 3:
        df_data = pd.DataFrame(data, columns=['X','y','conf1','conf2', 'conf3'])
        res = pg.partial_corr(data=df_data, x='X', y='y', covar=['conf1','conf2','conf3'])
    else: 
        raise ValueError('get_associations: not implemented for your number of confounds')
        
    return res, X
        
C = [age_sorted, epochs_rejected, sex_sorted] # confounds   
y = intel_sorted # RAPM scores

# lists to save results in
results_p_raw = []
results_p_adj = []
results_corr = []
results_mean =[]
results_std = []
results_min = []
results_max = []
results_names = []

# association between number of GFP peaks and RAPM scores
idx_cols_vars = np.where(df_complexity.columns == 'n_gfp_peaks')[0]
X = np.array(df_complexity.iloc[:,idx_cols_vars]).ravel()
res, X = get_association(X, y, C)

p = res['p-val'].to_numpy()
corr = res['r'].to_numpy()
results_p_raw.append(p)
results_p_adj.append(p) 
results_corr.append(corr)
results_mean.append(np.mean(X))
results_std.append(np.std(X))
results_min.append(np.min(X))
results_max.append(np.max(X))
results_names.append('n_gfp_peaks')


# associations between microstate measures and intelligence
n_MS = 5
n_measures = 4

#coverage
corr_ms = np.zeros((n_MS*n_measures))
p_ms = np.zeros((n_MS*n_measures))
cnt = 0
for k in range(n_MS):
           
    idx_cols_vars = np.where(df_complexity.columns == 'coverage_' + str(k))[0]
    X = np.array(df_complexity.iloc[:,idx_cols_vars]).ravel()
    res, X = get_association(X, y, C)

    p_ms[cnt] = res['p-val'].to_numpy()
    corr_ms[cnt] = res['r'].to_numpy()
    results_mean.append(np.mean(X))
    results_std.append(np.std(X))
    results_min.append(np.min(X))
    results_max.append(np.max(X))
    results_names.append('ms_' + 'coverage_' + str(k))
    cnt += 1
    
# lifespan
for k in range(n_MS):
           
    idx_cols_vars = np.where(df_complexity.columns == 'lifespan_' + str(k))[0]
    X = np.array(df_complexity.iloc[:,idx_cols_vars]).ravel()
    res, X = get_association(X, y, C)

    p_ms[cnt] = res['p-val'].to_numpy()
    corr_ms[cnt] = res['r'].to_numpy()
    results_mean.append(np.mean(X))
    results_std.append(np.std(X))
    results_min.append(np.min(X))
    results_max.append(np.max(X))
    results_names.append('ms_' + 'lifespan_' + str(k))
    cnt += 1
    
# lifespan GFP peaks
for k in range(n_MS):
           
    idx_cols_vars = np.where(df_complexity.columns == 'lifespan_peaks_' + str(k))[0]
    X = np.array(df_complexity.iloc[:,idx_cols_vars]).ravel()
    res, X = get_association(X, y, C)

    p_ms[cnt] = res['p-val'].to_numpy()
    corr_ms[cnt] = res['r'].to_numpy()
    results_mean.append(np.mean(X))
    results_std.append(np.std(X))
    results_min.append(np.min(X))
    results_max.append(np.max(X))
    results_names.append('ms_' + 'lifespan_peaks_' + str(k))
    cnt += 1
    
# frequency    
for k in range(n_MS):
           
    idx_cols_vars = np.where(df_complexity.columns == 'frequence_' + str(k))[0]
    X = np.array(df_complexity.iloc[:,idx_cols_vars]).ravel()
    res, X = get_association(X, y, C)

    p_ms[cnt] = res['p-val'].to_numpy()
    corr_ms[cnt] = res['r'].to_numpy()
    results_mean.append(np.mean(X))
    results_std.append(np.std(X))
    results_min.append(np.min(X))
    results_max.append(np.max(X))
    results_names.append('ms_' + 'frequence_' + str(k))
    cnt += 1

   
reject, p_val_adj = fdrcorrection(p_ms.flatten().T, alpha=0.05)    
results_corr.append(corr_ms.flatten().T)
results_p_raw.append(p_ms.flatten())
results_p_adj.append(p_val_adj)  


# associations between transition probabilities of microstates and RAPM scores
corr_ms1_ms2 = np.zeros((n_MS,n_MS))
p_ms1_ms2 = np.zeros((n_MS,n_MS))
for ms1 in range(n_MS):
    for ms2 in range(n_MS):
        
        idx_cols_vars = np.where(df_complexity.columns == 'transition_probability_'+ str(ms1) + '_' + str(ms2))[0]
        X = np.array(df_complexity.iloc[:,idx_cols_vars]).ravel()
        res, X = get_association(X, y, C)
        
        p_ms1_ms2[ms1,ms2] = res['p-val'].to_numpy()
        corr_ms1_ms2[ms1,ms2] = res['r'].to_numpy()
        results_mean.append(np.mean(X))
        results_std.append(np.std(X))
        results_min.append(np.min(X))
        results_max.append(np.max(X))
        results_names.append('ms_transition_' + str(ms1) + '_to_' + str(ms2))
        

reject, p_val_adj = fdrcorrection(p_ms1_ms2.flatten(), alpha=0.05)    
results_corr.append(corr_ms1_ms2.flatten())
results_p_raw.append(p_ms1_ms2.flatten())
results_p_adj.append(p_val_adj)             


# associations between transition probabilities of microstates GFP points only and RAPM scores
corr_ms1_ms2 = np.zeros((n_MS,n_MS))
p_ms1_ms2 = np.zeros((n_MS,n_MS))
for ms1 in range(n_MS):
    for ms2 in range(n_MS):
           
        idx_cols_vars = np.where(df_complexity.columns == 'transition_probability_peaks_'+ str(ms1) + '_' + str(ms2))[0]
        X = np.array(df_complexity.iloc[:,idx_cols_vars]).ravel()
        res, X = get_association(X, y, C)
        
        p_ms1_ms2[ms1,ms2] = res['p-val'].to_numpy()
        corr_ms1_ms2[ms1,ms2] = res['r'].to_numpy()
        results_mean.append(np.mean(X))
        results_std.append(np.std(X))
        results_min.append(np.min(X))
        results_max.append(np.max(X))
        results_names.append('ms_transition_peak_' + str(ms1) + '_to_' + str(ms2))
        

reject, p_val_adj = fdrcorrection(p_ms1_ms2.flatten(), alpha=0.05)    
results_corr.append(corr_ms1_ms2.flatten())
results_p_raw.append(p_ms1_ms2.flatten())
results_p_adj.append(p_val_adj)   


# association between Shannon entropy of GFP signals and RAPM scores
idx_cols_vars = np.where(df_complexity.columns == 'shannon_gfp')[0]
X = np.array(df_complexity.iloc[:,idx_cols_vars]).ravel()
res, X = get_association(X, y, C)
p = res['p-val'].to_numpy()
corr= res['r'].to_numpy()
results_p_raw.append(p)
results_p_adj.append(p) 
results_corr.append(corr)
results_mean.append(np.mean(X))
results_std.append(np.std(X))
results_min.append(np.min(X))
results_max.append(np.max(X))
results_names.append('shannon_gfp')

# association betweem Fuzzy entropy of GFP signals and RAPM scores
idx_cols_vars = np.where(df_complexity.columns == 'fuzzy_gfp')[0]
X = np.array(df_complexity.iloc[:,idx_cols_vars]).ravel()
res, X = get_association(X, y, C)
p = res['p-val'].to_numpy()
corr= res['r'].to_numpy()
results_p_raw.append(p)
results_p_adj.append(p) 
results_corr.append(corr)
results_mean.append(np.mean(X))
results_std.append(np.std(X))
results_min.append(np.min(X))
results_max.append(np.max(X))
results_names.append('fuzzy_gfp')


# assosiation between Shannon entropy of each channel and RAPM scores
corr_ch = np.zeros((28,1))
p_ch = np.zeros((28,1))
for ch in range(28):
           
    idx_cols_vars = np.where(df_complexity.columns == 'shannon_ch_' + str(channel_names[ch]))[0]
    X = np.array(df_complexity.iloc[:,idx_cols_vars]).ravel()
    res, X = get_association(X, y, C)
    p_ch[ch] = res['p-val'].to_numpy()
    corr_ch[ch] = res['r'].to_numpy()
    results_mean.append(np.mean(X))
    results_std.append(np.std(X))
    results_min.append(np.min(X))
    results_max.append(np.max(X))
    results_names.append('shannon_' + channel_names[ch])
    
reject, p_val_adj = fdrcorrection(p_ch.flatten(), alpha=0.05)    
results_corr.append(corr_ch.flatten())
results_p_raw.append(p_ch.flatten())
results_p_adj.append(p_val_adj) 


# associations between Fuzzy entropy of each channel and RAPM scores
corr_ch = np.zeros((28,1))
p_ch = np.zeros((28,1))
for ch in range(28):
           
    idx_cols_vars = np.where(df_complexity.columns == 'fuzzy_ch_' + str(channel_names[ch]))[0]
    X = np.array(df_complexity.iloc[:,idx_cols_vars]).ravel()
    res, X = get_association(X, y, C)
    p_ch[ch] = res['p-val'].to_numpy()
    corr_ch[ch] = res['r'].to_numpy()
    results_mean.append(np.mean(X))
    results_std.append(np.std(X))
    results_min.append(np.min(X))
    results_max.append(np.max(X))
    results_names.append('fuzzy_' + channel_names[ch])
    
reject, p_val_adj = fdrcorrection(p_ch.flatten(), alpha=0.05)    
results_corr.append(corr_ch.flatten())
results_p_raw.append(p_ch.flatten())
results_p_adj.append(p_val_adj)     

        
# associations between MSE and RAPM scores
corr_ch_sc = np.zeros((28,20))
p_ch_sc = np.zeros((28,20))
for ch in range(28):
    for sc in range(20):
            
        idx_cols_vars = np.where(df_complexity.columns == 'MSE_'+ channel_names[ch] + '_' + str(sc))[0]
        X = np.array(df_complexity.iloc[:,idx_cols_vars]).ravel()
        res, X = get_association(X, y, C)
        
        p_ch_sc[ch,sc] = res['p-val'].to_numpy()
        corr_ch_sc[ch, sc] = res['r'].to_numpy()

        results_mean.append(np.mean(X))
        results_std.append(np.std(X))
        results_min.append(np.min(X))
        results_max.append(np.max(X))
        results_names.append('mse_ch_' + channel_names[ch] + '_sc_' + str(sc))
        


results_corr.append(corr_ch_sc.flatten())
results_p_raw.append(p_ch_sc.flatten())

# measure clustersize of significant correlations
signi_ch_sc = np.zeros((28,20))
signi_ch_sc[np.where(p_ch_sc < 0.05)] = 1
lw_real, num_real = measurements.label(signi_ch_sc)
area_real = measurements.sum(signi_ch_sc, lw_real, index=np.arange(1,lw_real.max() + 1))

# permutation test for computing significant clustersize threshold        
areas_all = []
n_permutations = 100
for p in range(n_permutations):
    
    corr_ch_sc = np.zeros((28,20))
    p_ch_sc = np.zeros((28,20))
    intel_sorted_perm = np.random.permutation(intel_sorted) # permuted RAPM scores
    
    for ch in range(28):
        for sc in range(20):
            
            
            idx_cols_vars = np.where(df_complexity.columns == 'MSE_'+ channel_names[ch] + '_' + str(sc))[0]
            X = np.array(df_complexity.iloc[:,idx_cols_vars]).ravel()
            res, X = get_association(X, intel_sorted_perm, C)
  
            
            p_ch_sc[ch,sc] = res['p-val'].to_numpy()
            corr_ch_sc[ch, sc] = res['r'].to_numpy()

    signi_ch_sc = np.zeros((28,20))
    signi_ch_sc[np.where(p_ch_sc < 0.05)] = 1
    lw, num = measurements.label(signi_ch_sc)
    area = measurements.sum(signi_ch_sc, lw, index=np.arange(1,lw.max() + 1))
    areas_all.append(area)

# cluster sizes of permutated data
size_cluster_perm = np.concatenate(areas_all).astype(int)
count_cluster_sizes = np.bincount(size_cluster_perm)

CI95 = np.percentile(size_cluster_perm, 95) # 95th percentile of cluster sizes of null models

# compute p-values for each clustersize of real data
p_cluster_size = []
for n in range(1,int(area_real.max())+1):
    permute = size_cluster_perm
    real = n
    sum_bad = np.sum(permute >= real) # instances of null models have larger clustersize than real data
    p = sum_bad/permute.shape[0]
    p_cluster_size.append(p)
    
p_cluster_size = np.array(p_cluster_size)

# assign adjusted p-values to clusters of real data
p_adj = np.ones(p_ch_sc.shape)    
for c in range(num_real):
    
    c_idx = c + 1
    area_c = area_real[c]
    p_c = p_cluster_size[int(area_c)-1]
    p_adj[np.where(lw_real==c_idx)]=p_c
    
p_val_adj = np.array(p_adj).flatten()
results_p_adj.append(p_val_adj)



# associations between clustered MSE and RAPM scores
n_cluster = 8
n_scale = 4
corr_ch_sc = np.zeros((n_cluster,n_scale))
p_ch_sc = np.zeros((n_cluster,n_scale))
for ch in range(n_cluster):
    for sc in range(n_scale):
            
        idx_cols_vars = np.where(df_complexity.columns == 'MSE_cluster_channel_'+ str(ch) + '_scales_' + str(sc))[0]
        X = np.array(df_complexity.iloc[:,idx_cols_vars]).ravel()
        res, X = get_association(X, y, C)
        
        p_ch_sc[ch,sc] = res['p-val'].to_numpy()
        corr_ch_sc[ch, sc] = res['r'].to_numpy()
        
        
        results_mean.append(np.mean(X))
        results_std.append(np.std(X))
        results_min.append(np.min(X))
        results_max.append(np.max(X))
        results_names.append('mse_clustered_cluster_' + str(ch) + '_' + str(sc))
        

results_corr.append(corr_ch_sc.flatten())
results_p_raw.append(p_ch_sc.flatten())
reject, p_val_adj = fdrcorrection(p_ch_sc.flatten(), alpha=0.05)    
results_p_adj.append(p_val_adj)  


# associations between MSE of GFP and RAPM scores
rho_sc = np.zeros((20))
p_sc = np.zeros((20))
for sc in range(20):
            
    idx_cols_vars = np.where(df_complexity.columns == 'MSE_gfp_scale' + str(sc))[0]
    X = np.array(df_complexity.iloc[:,idx_cols_vars]).ravel()
    res, X = get_association(X, y, C)
    
    p_sc[sc] = res['p-val'].to_numpy()
    rho_sc[sc] = res['r'].to_numpy()
    results_mean.append(np.mean(X))
    results_std.append(np.std(X))
    results_min.append(np.min(X))
    results_max.append(np.max(X))
    results_names.append('mse_gfp_scale_' + str(sc))
    
reject, p_val_adj = fdrcorrection(p_sc.flatten(), alpha=0.05)    
results_corr.append(rho_sc.flatten())
results_p_raw.append(p_sc.flatten())
results_p_adj.append(p_val_adj)     


res1 = np.concatenate(results_corr)
res2 = np.concatenate(results_p_raw)
res3 = np.concatenate(results_p_adj)
res4 = np.array(results_mean)
res5 = np.array(results_std)
res6 = np.array(results_min)
res7 = np.array(results_max)

data = np.vstack((res1, res2, res3, res4, res5, res6, res7))
df_results_main = pd.DataFrame(data, columns = np.array(results_names), index = ['corr','p','p_adj','M','SD', 'min','max'])

df_results_main.to_pickle('df_results_repli')

#%% Prepare variables for prediction models

df_vars_prediction = df_complexity.copy()

# remove columns not needed for predicition
idx_mse = []
for ch in range(28):
    for sc in range(20):
            
        idx_mse.append(np.where(df_complexity.columns == 'MSE_'+ channel_names[ch] + '_' + str(sc))[0])
idx_mse = np.array(idx_mse).ravel()

idx_ID = np.where(df_complexity.columns == 'IDs_subjects_neuro')[0] 
idx_epochs = np.where(df_complexity.columns == 'epochs_rejected')[0] 


idx_to_remove = np.sort(np.concatenate([idx_mse, idx_ID, idx_epochs]))[::-1]

df_vars_prediction = df_vars_prediction.drop(df_vars_prediction.columns[idx_to_remove], axis = 1) 


feature_names = list(df_vars_prediction.columns) # names of variables for the prediction model
mat_var = np.array(df_vars_prediction) # all variables for the prediction model
confounds = np.vstack((age_sorted, epochs_rejected, sex_sorted)).T # confounds

#%% Prediction of intelligence with features selected in the main sample

def regress_confounds(feature, confounds):
    X = confounds
    y = feature
    regr = linear_model.LinearRegression()
    regr.fit(X, y)
    prediction = regr.predict(X)
    residuals = (y - prediction)
    residuals = scipy.stats.zscore(residuals)
    
    return residuals


# load features from main sample and find their indexes
names_features_pos = list(np.load('features_pos_main.npy')) # features positively correlating with intelligence in main sample
names_features_neg = list(np.load('features_neg_main.npy')) # features negatively correlating with intelligence in main sample

# get indexes of features positively correlating with intelligence in main sample
ind_p_pos = []
for name_feature in names_features_pos:
    
    ind_p_pos.append(np.where(df_vars_prediction.columns == name_feature)[0])
ind_p_pos = np.array(ind_p_pos).ravel()

# get indexes of features negatively correlating with intelligence in main sample
ind_p_neg = []
for name_feature in names_features_neg:
    
    ind_p_neg.append(np.where(df_vars_prediction.columns == name_feature)[0])
ind_p_neg = np.array(ind_p_neg).ravel()


# controlling for confounds (via linear regression)
intelligence_reg = regress_confounds(intel_sorted, confounds)

vars_reg = []
for el in range(mat_var.shape[1]):
    
    res = regress_confounds(mat_var[:,el], confounds)
    vars_reg.append(res)

X = np.array(vars_reg).T
y = intelligence_reg   

X_sel_neg = X[:,ind_p_neg] # variables of replication sample that are negatively correlated with intelligence in main sample
X_sel_pos = X[:,ind_p_pos] # variables of replication sample that are positively correlated with intelligence in main sample


y_pred_neg = np.mean(X_sel_neg, axis = 1) # mean of negatively correlating variables (p<0.05)
y_pred_pos = np.mean(X_sel_pos, axis = 1) # mean of positively correlating variables (p<0.05)

y_pred = y_pred_pos - y_pred_neg 

corr_repli = scipy.stats.pearsonr(y_pred,y)

print('---------------------------------------')
print('Model 3: Replication sample, acurracy: corr')
print(str(corr_repli[0]))
print('---------------------------------------')

# visualize y vs y_pred
plt.figure('model repli')
ax = sns.regplot(x=y, y=y_pred, ci = None, color = 'black', marker = '.')
plt.xlabel('true intelligence score', fontsize = '16')
plt.ylabel('predictied intelligence score', fontsize = '16') 
plt.yticks(fontsize = '16')
plt.xticks(fontsize = '16')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig('model_repli.jpg', format='jpg', dpi = 1200, bbox_inches='tight')

#%% Permutation test (null models)

corr_perm = []
for n in range(1000):
    
    y_p = np.random.permutation(intel_sorted)

    intelligence_reg = regress_confounds(y_p, confounds)
    vars_reg = []
    for el in range(mat_var.shape[1]):
        
        res = regress_confounds(mat_var[:,el], confounds)
        vars_reg.append(res)
        
    X = np.array(vars_reg).T
    y = intelligence_reg   
    
    X_sel_neg=X[:,ind_p_neg] # variables of replication sample that are negatively correlated with intelligence in main sample
    X_sel_pos=X[:,ind_p_pos] # variables of replication sample that are positively correlated with intelligence in main sample


    y_pred_neg = np.mean(X_sel_neg, axis = 1) # mean of negatively correlating variables
    y_pred_pos = np.mean(X_sel_pos, axis = 1) # mean of positively correlating variables

    y_pred = y_pred_pos - y_pred_neg 

    corr_perm.append(scipy.stats.pearsonr(y_pred,y)[0])
   
# 95th percentile of model performance values of null models  
CI95 = np.percentile(np.array(corr_perm), 95)

# p-value for prediction in replication sample
real = corr_repli[0]
permute = np.array(corr_perm) 
sum_bad = np.sum(permute >= real) # sum of permutations where null model outperforms real model
p = sum_bad/permute.shape[0]

# visualize histogram of null model accuracies vs accuracy of true model
plt.figure()
ax=sns.histplot(permute, color = 'gray', alpha = 0.5)
plt.axvline(real)
plt.ylabel('frequency', fontsize = '16') 
plt.xlabel('corr', fontsize = '16') 
plt.yticks(fontsize = '16')
plt.xticks(fontsize = '16')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.locator_params(axis='y', nbins=5)
plt.locator_params(axis='x', nbins=5)
plt.savefig('hist_repli_permutation_test.jpg', format='jpg', dpi = 1200, bbox_inches='tight')


