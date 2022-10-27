# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 09:56:55 2022

@author: Jonas A. Thiele
"""

#%% Imports

import os
import numpy as np
import mne
import neurokit2 as nk
import matplotlib.pyplot as plt
from pyprep.find_noisy_channels import NoisyChannels # part of the PREP pipeline for detecting 
# bad channels https://doi.org/10.3389/fninf.2015.00016
import entropy #https://github.com/ixjlyons/entro-py/tree/b371cbc5f72197ca9f59506266de8cb927a2f0d8
import copy
from scipy.signal import find_peaks
import microstates_subject
import microstates_group
import pandas as pd
import pickle
#%% Preprocess data 1 

# current work directory
cwd = os.getcwd()

# file path with all the eeg data
filedir = os.path.join(cwd,"Fopsi_all")

# read names of files with specific file extension (vhdr) and of a specific scan condition (EC)
filesEC = [filedir +'\\'+ f for f in os.listdir(filedir) 
         if all(x in f for x in ("vhdr", "Rest", "EC"))]


# initializing lists for storing IDs, icas, preprocessed EEG data 
IDs_subjects = []
all_ica=[]
eeg_all=[]  

# loop over all subjects (filenames saved in filesEC)
for f in filesEC: 
     
    fname = f.replace('\\','/') # replacing backslash with slash because mne read cannot read \\
   
    ID = fname.split('/')[-1].split('_')[0]
    IDs_subjects.append(ID)         
    raw = mne.io.brainvision.read_raw_brainvision(fname, preload=True) # read file as mne object
    
    # remap non-eeg channels
    raw.set_channel_types(mapping={'hore 28': 'eog', 'vou 17': 'eog', 'holi 32': 'eog' ,'mast2':'misc'})
    
    # drop non-EEG channels
    raw.drop_channels(['vou 17','hore 28','holi 32','mast2']) 
    
    
    # load and set the montage (1020 standard)
    ten_twenty_montage = mne.channels.make_standard_montage('standard_1020')
    raw_temp = raw.copy().set_montage(ten_twenty_montage)
    
    # filtering
    raw_temp.filter(0.1, None, n_jobs=1, method="iir",)  
    raw_temp.filter(None, 40, n_jobs=1, method="iir",) 
    
    # resampling
    freq_resample = 250;
    raw_temp = raw_temp.resample(sfreq=freq_resample)
    
    
    # initializing the object NoisyChannels which has different functions for detecting bad channels
    # it is part of the PREP pipeline (see imports for reference)
    noisy_detector = NoisyChannels(raw_temp)
    
    # looking for NaNs (not a number) values in channels and for channels with flat signal
    noisy_detector.find_bad_by_nan_flat() 
    
    # looking for channels with z-threshold larger deviation_threshold
    noisy_detector.find_bad_by_deviation(deviation_threshold=5.0)
    
    # looking for channels that are too low correlated with other channels 
    noisy_detector.find_bad_by_correlation()
    
    # collect unique detected bad channels
    bad_channels = set(noisy_detector.bad_by_nan + noisy_detector.bad_by_flat
                       + noisy_detector.bad_by_deviation + noisy_detector.bad_by_correlation
                       + noisy_detector.bad_by_hf_noise)
    
    # mark detected bad channels as information for mne raw object
    raw_temp.info["bads"] = list(bad_channels)
    
    # interpolate bad channels
    raw_temp.interpolate_bads()
    
    # apply average reference on scalp electrodes
    raw_temp.set_eeg_reference(ref_channels = 'average')
    
    # ICA
    ica = mne.preprocessing.ICA(n_components=20).fit(raw_temp)
              
    all_ica.append(ica)
    eeg_all.append(raw_temp)

#%% Preprocess data 2   

# manually (visually) identify EOG artefacts from the ICA components of the first 10 subjects and collect them in bads_all
# after identification, bad components need to be changed in the following code!
for i in range(10):
    all_ica[i].plot_components(outlines="skirt", title=str(IDs_subjects[i]))

bads_all = []

bads = [0,1]
bad_0 = all_ica[0].get_components()[:,bads]
bads_all.append(bad_0)

bads = [0,1]
bad_1 = all_ica[1].get_components()[:,bads]
bads_all.append(bad_1)

bads = [0,1]
bad_2 = all_ica[2].get_components()[:,bads]
bads_all.append(bad_2)

bads = [0,1,2]
bad_3 = all_ica[3].get_components()[:,bads]
bads_all.append(bad_3)

bads = [1,2]
bad_4 = all_ica[4].get_components()[:,bads]
bads_all.append(bad_4)

bads = [0,2]
bad_5 = all_ica[5].get_components()[:,bads]
bads_all.append(bad_5)

bads = [0,1,2,3,4]
bad_6 = all_ica[6].get_components()[:,bads]
bads_all.append(bad_6)

bads = [0,1,2,8]
bad_7 = all_ica[7].get_components()[:,bads]
bads_all.append(bad_7)

bads = [0,1]
bad_8 = all_ica[8].get_components()[:,bads]
bads_all.append(bad_8)

bads = [0,1,2,4,8]
bad_9 = all_ica[9].get_components()[:,bads]
bads_all.append(bad_9)

bads_all = np.concatenate(bads_all,axis=1)

# plot determined bad components 
fig, axes = plt.subplots(nrows=5, ncols=6)
cnt=0
for axes_row in axes:
    
    for ax in axes_row:
        if cnt < bads_all.shape[1]:
            mne.viz.plot_topomap(bads_all[:,cnt], raw_temp.info, axes=ax, show=False)
            cnt +=1
fig.tight_layout()
plt.savefig('bad_ica_components_main.jpg', format='jpg', dpi = 1200)

# find components similar to determined bad compnetns (corrmap) in all subjects' ICAs 
icas = copy.deepcopy(all_ica)
for ii, ica_template in enumerate(bads_all.T):
    mne.preprocessing.corrmap(icas, ica_template,
                              threshold=0.8,
                              label="eog" + str(ii), show=False, plot=False)

# collecting all found bad components across subjects (only for evaluation purposes)
list_cnts = []    
list_bad_all = []    
for ica in icas:
    cnt = 0
    for bads in list(set(np.hstack(ica.labels_.values()).astype(int))):
        cnt = cnt+1
        list_bad_all.append(ica.get_components()[:,bads])
    print(cnt)
    list_cnts.append(cnt)

# exclude found bad components for all subjects
data_all = copy.deepcopy(eeg_all)
for ica, data in zip(icas, data_all):
    print(list(set(np.hstack(ica.labels_.values()).astype(int))))
    
    ica.exclude = list(set(np.hstack(ica.labels_.values()).astype(int)))
    ica.apply(data)  

# drop bad epochs, exclude subjects with high number of dropped epochs    
data_epoched_all = []  
IDs_subjects_neuro = [] # subject IDs of all analyzed subjects (without excluded subjects)
epochs_rejected_all = [] 
data_concat_all = []
for s in range(len(data_all)):
    
    raw_ica = data_all[s]
    
    data = raw_ica.get_data()
    
    # define range from 10 to 310 seconds
    data = data[:,10*freq_resample:310*freq_resample]
    
    # window size
    step = 2*freq_resample;

    # looking for bad epochs that have at least one value higher than max value
    eeg_thresh_max = 100e-6
    
    # excludes all epochs in which at least on value exceeds the threshold and 
    # returns all other values in an array
    data_epoched = np.asarray([d_ for d_ in (data[:, first:first+step]  
                                for first in range(0, data.shape[1], step))
                  if d_.shape[1] == step and np.abs(d_).max() < eeg_thresh_max 
                  ])
    
    # collect epoched data
    data_epoched_all.append(data_epoched)
    
    # reshape epoched data array to concatenated epochs 
    data_concat = data_epoched.transpose(1,0,2).reshape(data_epoched.shape[1],-1)
    
    no_bad_epochs = data.shape[1]/step - data_epoched.shape[0]
    
    # subjects that survived the exclusion criteria (<1/3 of 150 epochs bad)
    if np.array(no_bad_epochs) < 50:
        IDs_subjects_neuro.append(IDs_subjects[s])
        data_concat_all.append(data_concat)
        epochs_rejected_all.append(no_bad_epochs)
        
# get number of bad epochs for every subject    
no_bad_epochs_all = []
for d in data_epoched_all:
    
    no_bad_epochs_all.append(data.shape[1]/step - d.shape[0])
    
no_bad_epochs_all = np.array(no_bad_epochs_all)        

idx_bad_epochs = np.where(np.array(no_bad_epochs_all) >= 50)[0] # indexes of excluded subjects (>50 excluded epochs)

#%% Compute microstate measures

gfp_peaks_all = []
n_gfp_peaks_all = []
maps_subject_all = []
for data_concat in data_concat_all:

    # compute GFP and peaks of GFP
    gfp = np.std(data_concat, axis=0)
    peaks, _ = find_peaks(gfp, distance=2)
    gfp_peaks_all.append(peaks)
    n_peaks = len(peaks)
    n_gfp_peaks_all.append(n_peaks)

    
    # get subject-specific microstates
    # segment the data into 5 microstates that maximize GEV (GEV = global explained variance = percentage of total variance explained by a given microstate)
    # only time points that are GFP peaks are used (within microstates_subject function)
    maps, segmentation = microstates_subject.segment(data_concat, n_states=5, n_inits=1000, thresh=1e-10, max_iter=10000)
    maps_subject_all.append(maps)
    
maps_subjects_all_arr = np.array(maps_subject_all)

# reshape for clustering all subject-specific microstates  
maps_subjects_all_arr = np.reshape(maps_subjects_all_arr,((maps_subjects_all_arr.shape[0] * maps_subjects_all_arr.shape[1]), maps_subjects_all_arr.shape[2]))
maps_subjects_all_arr = maps_subjects_all_arr.T

# get group specific microstates
# segment the data into 5 microstates that maximize GEV (GEV = global explained variance = percentage of total variance explained by a given microstate)
maps_group, segmentation = microstates_group.segment(maps_subjects_all_arr, n_states=5, n_inits=1000, thresh=1e-10, max_iter=10000)
np.save('microstates_group.npy', maps_group) # plot group microstates 

#plot maps_group
microstates_group.plot_maps(maps_group, raw_temp.info)

# backfitting to whole eeg signal
maps_group = maps_group.T
map_sequence_all = []
corr_backmap = []
for data_concat in data_concat_all:
    
    corr_subj =  []
    for m in range(maps_group.shape[1]):
        map_corr = microstates_group._corr_vectors(data_concat, maps_group[:,m].reshape(-1,1)) 
        corr_subj.append(map_corr)
        
    corr_subj = np.array(corr_subj)
    corr_subj_abs = abs(corr_subj)
    corr_backmap.append(corr_subj_abs)
    map_sequence = np.argmax(corr_subj_abs, axis = 0)
    map_sequence_all.append(map_sequence)
  

# backfitting to GFP peaks of eeg signal only
map_sequence_peaks_all = []
for data_concat, gfp_peaks_s in zip(data_concat_all, gfp_peaks_all):
    corr_subj =  []
    
    for m in range(maps_group.shape[1]):
        map_corr_ind = microstates_group._corr_vectors(data_concat[:, gfp_peaks_s], maps_group[:,m].reshape(-1,1)) 
        corr_subj.append(map_corr_ind)
        
    corr_subj = np.array(corr_subj)
    corr_subj_abs = abs(corr_subj)
    map_sequence_peaks = np.argmax(corr_subj_abs, axis = 0)
    map_sequence_peaks_all.append(map_sequence_peaks)



# compute coverage time per microstate all time points
n_total_occurences_states_all = []
coverage = []
for seq in map_sequence_all:
    n_total_occurences_states_all.append(np.bincount(seq))
    coverage.append(np.bincount(seq)/len(seq))

# compute coverage time per microstate at GFP peaks    
n_total_occurences_states_peaks_all = []
coverage_peak = []
for seq in map_sequence_peaks_all:
    n_total_occurences_states_peaks_all.append(np.bincount(seq))   
    coverage_peak.append(np.bincount(seq)/len(seq))

    
# compute occurences of microstates (times it is transitioned into a microstate (no matter its duration))
# occurences at all time points
states_single_all = []
for st in map_sequence_all:
    st = np.array(st)
    diff_st = np.diff(st)
    pos = np.where(diff_st != 0)
    pos = pos[0] + 1 
    pos = np.insert(pos, 0, 0)
    states_single_all.append(st[pos])

n_single_occurences_states_all = []
for st in states_single_all:
    n_single_occurences_states_all.append(np.bincount(st))
    
# occurences at all GFP peaks  
states_single_peaks_all = []
for st in map_sequence_peaks_all:
    st = np.array(st)
    diff_st = np.diff(st)
    pos = np.where(diff_st != 0)
    pos = pos[0] + 1 
    pos = np.insert(pos, 0, 0)
    states_single_peaks_all.append(st[pos])

n_single_occurences_states_peaks_all = []
for st in states_single_peaks_all:
   n_single_occurences_states_peaks_all.append(np.bincount(st))   

# frequency of microstates   
frequency = []
for st, e in zip(n_single_occurences_states_all, epochs_rejected_all):
    freq = st / (150-e) / 2 # occurences divided by 2*epochs (1 epoch = 2 s)
    frequency.append(freq) 
    
    
# lifespan
lifespan = np.array(n_total_occurences_states_all)/np.array(n_single_occurences_states_all)

# lifespan at GFP peaks
lifespan_peaks = np.array(n_total_occurences_states_peaks_all)/np.array(n_single_occurences_states_peaks_all)


# Transition probabilities
# Function t_empirical used from:
# https://github.com/Frederic-vW/eeg_microstates/blob/78283c71fb82d80704ba954d29bf88b830f2e416/eeg_microstates3.py
# Copyright (c) 2017 Frederic von Wegner 
# MIT License

# transition matrix
def t_empirical(data, n_clusters):
    T = np.zeros((n_clusters, n_clusters))
    n = len(data)
    for i in range(n-1):
        T[data[i], data[i+1]] += 1.0
    p_row = np.sum(T, axis=1)
    for i in range(n_clusters):
        if ( p_row[i] != 0.0 ):
            for j in range(n_clusters):
                T[i,j] /= p_row[i]  # normalize row sums to 1.0
    return T


# transition matrix of microstates all time points
trans_mat = []
for seq in map_sequence_all:
    
    trans_mat.append(t_empirical(seq, 5))
    
# transition matrix of microstates GFP peaks only    
trans_mat_peak = []
for seq in map_sequence_peaks_all:
    
    trans_mat_peak.append(t_empirical(seq, 5))

       
#%% Compute entropies

# MSE on GFP signal
gfp_all= []
MSE_gfp_all = []
for data_concat in data_concat_all: 
     
    # calculate whole brain MSE on GFP
    gfp = np.std(data_concat, axis=0)
    gfp_all.append(gfp)
    MSE_gfp, info_MSE_gfp = nk.entropy_multiscale(gfp, show=False, scale=21)
    MSE_gfp_vals = info_MSE_gfp['Values']
    MSE_gfp_all.append(MSE_gfp_vals)
    

# MSE for each channel
MSE_all = []    
for data_concat in data_concat_all:
    MSE = []
    #loop over channels and calculate MSE
    for channel in range(28):
        
        MSE_auc, info_MSE = nk.entropy_multiscale(data_concat[channel,:], show=False, scale=21)
        MSE_vals = info_MSE['Values']
        MSE.append(MSE_vals)
        
    MSE_all.append(np.array(MSE))
    

# Fuzzy entropy of GFP
divident = 16
fuzzy_gfp_all = []
for s in range(len(gfp_all)):

    fe_all_div = []
    gfp_signal = gfp_all[s]
    
    len_signal = int(np.ceil(gfp_signal.shape[0]/divident))
    for div in range(divident):
        signal_div = gfp_signal[div*len_signal:div*len_signal+len_signal]
        fe_div = entropy.fuzzyen(signal_div, dim=2, r=0.2, n=1)
        fe_all_div.append(fe_div)
    
    fuzzy_gfp_all.append(np.mean(np.array(fe_all_div)))
     
# Fuzzy entropy all channels
fuzzy_all = []
for data_concat in data_concat_all:
    
    len_signal = int(np.ceil(data_concat.shape[1]/divident))
    
    fe_ch_all = []
    
    #loop over channels
    for channel in range(28):
        fe_ch = []
        
        for div in range(divident):
            signal_div = data_concat[channel,div*len_signal:div*len_signal+len_signal]
            fe = entropy.fuzzyen(signal_div, dim=2, r=0.2, n=1) 
            fe_ch.append(fe)
        fe_ch_avg = np.mean(np.array(fe_ch))    
        fe_ch_all.append(fe_ch_avg)
       
    fuzzy_all.append(np.array(fe_ch_all))
 

# Shannon entropy of GFP
shannon_gfp_all =[]
for gfp_s in gfp_all:
    
    data_rounded = np.around(gfp_s, decimals=7) 
    shannon_gfp = nk.entropy_shannon(data_rounded)
    shannon_gfp_all.append(shannon_gfp[0])
     

# Shannon entropy of each channel 
shannon_all = []
for data_concat in data_concat_all:
    
    data_rounded = np.around(data_concat, decimals=7)
    se_channel = []    
    # loop over channels and calculate se
    for channel in range(28):
        se = nk.entropy_shannon(data_rounded[channel, :]) 
        se_channel.append(se[0])
    
    shannon_all.append(np.array(se_channel))
 


#%% Post-hoc analyses microstate measures       
        
# similarity of subject specific microstates
similarity_subject_microstates_all = []
for maps_subject in maps_subject_all:
    
    correlation_maps_subject = abs(np.corrcoef(maps_subject))
    mean_correlation_maps_subject = np.mean(correlation_maps_subject)
    similarity_subject_microstates_all.append(mean_correlation_maps_subject)

# explained variance of individual signals by subject-specific microstates
gev_subject_all = []
for data_concat, maps_subject in zip(data_concat_all, maps_subject_all):

    activation = maps_subject.dot(data_concat)
    segmentation = np.argmax(np.abs(activation), axis=0)
    map_corr = microstates_group._corr_vectors(data_concat, maps_subject[segmentation].T)
    gfp = np.std(data_concat, axis=0)
    gfp_sum_sq = np.sum(gfp ** 2)
    gev = sum((gfp * map_corr) ** 2) / gfp_sum_sq
    gev_subject_all.append(gev)


# explained variance of individual signals by group states
gev_group_all = []
for data_concat in data_concat_all:
    
    activation = maps_group.T.dot(data_concat)
    segmentation = np.argmax(np.abs(activation), axis=0)
    map_corr = microstates_group._corr_vectors(data_concat, maps_group.T[segmentation].T)
    gfp = np.std(data_concat, axis=0)
    gfp_sum_sq = np.sum(gfp ** 2)
    gev = sum((gfp * map_corr) ** 2) / gfp_sum_sq
    gev_group_all.append(gev)
        

#%% Save variables in tables

channel_names = eeg_all[0].info.ch_names

# number of GFP peaks
variables = np.array(n_gfp_peaks_all)
table_n_gfp_peaks = pd.DataFrame(variables, columns = ['n_gfp_peaks'])

# coverage of micorsates
variables = np.array(coverage)
names = []
for i in range(variables.shape[1]):
    names.append('coverage_' + str(i))    

names = np.array(names)
table_coverage = pd.DataFrame(variables, columns = names)

# lifespan of microstates
variables = np.array(lifespan)
names = []
for i in range(variables.shape[1]):
    names.append('lifespan_' + str(i))    

names = np.array(names)
table_lifespan = pd.DataFrame(variables, columns = names)


# lifespan of microstates GFP peaks only
variables = np.array(lifespan_peaks)
names = []
for i in range(variables.shape[1]):
    names.append('lifespan_peaks_' + str(i))    

names = np.array(names)
table_lifespan_peaks = pd.DataFrame(variables, columns = names)

# frequency of microstates
variables = np.array(frequency)
names = []
for i in range(variables.shape[1]):
    names.append('frequence_' + str(i))    

names = np.array(names)
table_frequence = pd.DataFrame(variables, columns = names)

# transition probabilities of microstates
variables = np.array(trans_mat)
column_names = []
for ms1 in range(variables.shape[1]):
    names = []
    for ms2 in range(variables.shape[2]):
        names.append('transition_probability_'+ str(ms1) + '_' + str(ms2))
    column_names.append(np.array(names))  
column_names = np.array(column_names)

data = np.reshape(variables,(variables.shape[0], variables.shape[1]*variables.shape[2]))
columns = np.reshape(column_names,(variables.shape[1]*variables.shape[2]))
table_transmat = pd.DataFrame(data, columns = columns)


# transition probabilities of microstates GFP peaks only
variables = np.array(trans_mat_peak)
column_names = []
for ms1 in range(variables.shape[1]):
    names = []
    for ms2 in range(variables.shape[2]):
        names.append('transition_probability_peaks_'+ str(ms1) + '_' + str(ms2))
    column_names.append(np.array(names))  
column_names = np.array(column_names)

data = np.reshape(variables,(variables.shape[0], variables.shape[1]*variables.shape[2]))
columns = np.reshape(column_names,(variables.shape[1]*variables.shape[2]))
table_transmat_peak = pd.DataFrame(data, columns = columns)

# Shannon entropy of GFP
variables = np.array(shannon_gfp_all)
table_shannon_gfp = pd.DataFrame(variables, columns = ['shannon_gfp'])

# Fuzzy entropy of GFP
variables = np.array(fuzzy_gfp_all)
table_fuzzy_gfp = pd.DataFrame(variables, columns = ['fuzzy_gfp'])


# Shannon entropy of all channels
variables = np.array(shannon_all)
names = []
for i in range(variables.shape[1]):
    names.append('shannon_ch_' + str(channel_names[i]))    
names = np.array(names)
table_shannon = pd.DataFrame(variables, columns = names)

# Fuzzy entropy of all channels
variables = np.array(fuzzy_all)
names = []
for i in range(variables.shape[1]):
    names.append('fuzzy_ch_' + str(channel_names[i]))    
names = np.array(names)
table_fuzzy = pd.DataFrame(variables, columns = names)


# MSE all channels all scales
variables = np.array(MSE_all)
column_names = []
for ch in range(variables.shape[1]):
    names = []
    for sc in range(variables.shape[2]):
        names.append('MSE_'+ channel_names[ch] + '_' + str(sc))
    column_names.append(np.array(names))  
column_names = np.array(column_names)

data = np.reshape(variables,(variables.shape[0], variables.shape[1]*variables.shape[2]))
columns = np.reshape(column_names,(variables.shape[1]*variables.shape[2]))
table_MSE = pd.DataFrame(data, columns = columns)

       
# MSE in spatial and temporal cluster for prediction models
spatial_cluster = [['Fp1','Fp2'],['FC1','FC2','FC5','FC6'],['F3','F7','Fz','F4','F8'],['T7','T8'],['CP5','CP1','CP6','CP2'],['C3','Cz','C4'],['P3','P7','P4','P8','Pz'],['O1','Oz','O2']]
scale_cluster = np.arange(0,5), np.arange(5,10), np.arange(10,15), np.arange(15,20) # 4 clusters with 5 consecutive time steps each

idx_spatial_cluster = []
# get indexes for channels of spatial clusters
for c in spatial_cluster:
    ind_c = []
    for e in c:
        
        ind_c.append(channel_names.index(e))
    
    idx_spatial_cluster.append(np.array(ind_c))


MSE_cluster = []
for s in range(len(MSE_all)):
    # averaging MSE within each spatial and temporal cluster
    MSE_s = MSE_all[s]
    MSE_cluster_s = []
    for idx in idx_spatial_cluster:
        MSE_cluster_ch =[]
        for sc in scale_cluster:
            Mx = MSE_s[idx,:]
            Mx = Mx[:,sc]
            MSE_cluster_ch.append(np.mean(Mx))
        
        MSE_cluster_s.append(np.array(MSE_cluster_ch))
    MSE_cluster.append(np.array(MSE_cluster_s))
    
    
variables = np.array(MSE_cluster)
column_names = []
for ch in range(variables.shape[1]):
    names = []
    for sc in range(variables.shape[2]):
        names.append('MSE_cluster_channel_'+ str(ch) + '_scales_' + str(sc))
    column_names.append(np.array(names))  
column_names = np.array(column_names)

data = np.reshape(variables,(variables.shape[0], variables.shape[1]*variables.shape[2]))
columns = np.reshape(column_names,(variables.shape[1]*variables.shape[2]))
table_MSE_cluster = pd.DataFrame(data, columns = columns)

# MSE of GFP
variables = np.array(MSE_gfp_all)
names = []
for sc in range(variables.shape[1]):
    names.append('MSE_gfp_scale' + str(sc))    
names = np.array(names)
table_MSE_gfp = pd.DataFrame(variables, columns = names)

# Post-hoc analyses
variable_1 = np.array(similarity_subject_microstates_all)
variable_2 = np.array(gev_subject_all)
variable_3 = np.array(gev_group_all)
variables = np.vstack((variable_1, variable_2, variable_3)).T
table_posthoc = pd.DataFrame(variables, columns = ['similarity_subject_micosates','gev_subject','gev_group'])



df_complexity = pd.concat([table_n_gfp_peaks, table_coverage, 
                                       table_lifespan, table_lifespan_peaks, 
                                       table_frequence, table_transmat, table_transmat_peak, 
                                       table_shannon_gfp, table_fuzzy_gfp, table_shannon,
                                       table_fuzzy, table_MSE, table_MSE_cluster, 
                                       table_MSE_gfp, table_posthoc], axis = 1)

df_complexity.insert(0, 'epochs_rejected', np.array(epochs_rejected_all))
df_complexity.insert(0, 'IDs_subjects_neuro', np.array(IDs_subjects_neuro))

df_complexity.to_pickle("./df_complexity.pkl")

with open("channel_names", "wb") as fp:   
    pickle.dump(channel_names, fp)
    
np.savetxt('channel_names.csv', np.array(channel_names), delimiter=',', fmt='%s')

