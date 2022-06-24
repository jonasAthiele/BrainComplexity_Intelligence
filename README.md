# BrainComplexity_Intelligence


## 1. Scope
The repository contains scripts for the analyses used in the paper **"Multimodal Brain Signal Complexity Predicts Human Intelligence"** coauthored by Jonas A. Thiele, Aylin Richter, and Kirsten Hilger (doi: added once published). Herein, the relations between intelligence and different measures of brain signal complexity (assessed from resting-state EEG) were analyzed.
The scripts in this repository can be used to replicate the analyses of the paper or more generally, to study associations between individual differences (e.g., intelligence) and measures of brain signal complexity.
In case you have questions or trouble with the scripts, feel free to contact me at jonas.thiele@uni-wuerzburg.de
## 2. Data
Raw data for this study consisted of: Resting-state EEG data (5 minutes eyes-closed), intelligence scores (Raven's Advanced Progressive Matrices scores, Raven and Court, 1998), and behavioral data (Age, Sex). 
Raw data of the main sample was acquired at Würzburg University while raw data of the replication sample was acquired at Frankfurt University.
The raw data analyzed during the study will be made available by the corresponding author upon reasonable request.

## 3. Structure and script description
### Main sample

For the analyses conducted in this study, the scripts should be run in the following order:

1.	`get_complexity_main` - Main sample: Preprocessing, computation of complexity measures (entropy and microstate measures).
                            Note that ICA components wanting to be excluded for artefact removal during preprocessing need to be adapted manually (chosen by visual inspection).
                            Preprocessing is mainly based on MNE (Gramfort et al., 2013) and Pyprep (Bigdely-Shamlo et al., 2015). 
  
2.	`analysis_main` - Factor analysis, single associations between complexity measures and intelligence, and multimodal model to predict intelligence in main sample
  

### Replication sample

3.	`get_complexity_repli` - Replication sample: Preprocessing, computation of complexity measures (entropy and microstate measures).
                             Note that ICA components wanting to be excluded for artefact removal during preprocessing need to be adapted manually (chosen by visual inspection). 
                             Preprocessing is mainly based on MNE (Gramfort et al., 2013) and Pyprep (Bigdely-Shamlo et al., 2015).   
  
4.	`analysis_repli` - Factor analysis, single associations between complexity measures and intelligence, and multimodal model to predict intelligence in replication sample

### Additional analyses

5. `compare_mse` - Comparing patterns of associations between multiscale entropy and intelligence between main and replication sample

### Visualization

6. `plot_results` - Visualization of associations between complexity measures and intelligence


### Functions

- `microstates_subject` - Computing subject-specific microstates, functions from mne-microstates module, authored by Marijn van Vliet (https://github.com/wmvanvliet/mne_microstates.git)

- `microstates_group` - Computing group-specific microstates, same functions as `microstates_subject` with a minor change.

- `entropy` - For computation of Fuzzy entropy, authored by Kenneth Lyons (https://github.com/ixjlyons/entro-py.git)

Note: Licenses of these functions are provided in the `Functions` folder.

### Results

The `Results` folder contains:

- `df_complexity.pkl` - Complexity measures calculated in the main sample (output from `get_complexity_main`)

- `info_raw.fif` - EEG file with channel locations for plotting

- `microstates_group.npy` - Group microstates computed in the main sample (output from `df_complexity.pkl`, necessary for replication analyses)

- `channel_names` - Channel names of main sample (output from `df_complexity.pkl`)

- `df_complexity_repli.pkl` - Complexity measure calculated in the replication sample (output from  `get_complexity_repli`)

- `df_results_main` - Associations between complexity measures and intelligence from main sample (output from `analysis_main`)

- `df_results_repli` - Associations between complexity measures and intelligence from replication sample (output from `analysis_repli`)

## 4. Software requirements
Analyses were conducted using Python 3.8.

For requirements please refer to `requirements.txt`.	

## References
Bigdely-Shamlo N, Mullen T, Kothe C, Su K-M, Robbins KA (2015) The PREP pipeline: standardized preprocessing for large-scale EEG analysis. Front Neuroinform 9.

Gramfort A, Luessi M, Larson E, Engemann DA, Strohmeier D, Brodbeck C, Goj R, Jas M, Brooks T, Parkkonen L, Hämäläinen M (2013) MEG and EEG data analysis with MNE-Python. Front Neurosci 0:267.

Raven JC, Court JH (1998) Manual for Raven’s progressive matrices and vocabulary scales.
## Copyright
Copyright (cc) 2022 by Jonas Thiele

<a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-nd/4.0/88x31.png" /></a><br /><span xmlns:dct="http://purl.org/dc/terms/" property="dct:title">Files of BrainComplexity_Intelligence</span> by <a xmlns:cc="http://creativecommons.org/ns#" href="https://github.com/jonasAthiele/BrainReconfiguration_Intelligence" property="cc:attributionName" rel="cc:attributionURL">Jonas A. Thiele</a> are licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/">Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License</a>.

Note that external functions used may have other licenses.
