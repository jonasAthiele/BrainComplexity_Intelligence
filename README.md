# BrainComplexity_Intelligence


## 1. Scope
The repository contains scripts for the analyses used in the paper **"Multimodal Brain Signal Complexity Predicts Human Intelligence"** coauthored by Jonas A. Thiele, Aylin Richter, and Kirsten Hilger (doi: add when published). Herein, the relations between intelligence and different measures of brain signal complexity (assessed from resting-state EEG) are analyzed.
The scripts in this repository can be used to replicate the analyses of the paper or more generally, to study associations between individual differences (e.g., intelligence) and measures of brain signal complexity.
If you have questions or trouble with the scripts, feel free to contact me: jonas.thiele@uni-wuerzburg.de
## 2. Data
Raw data included: Resting-state EEG data (5 minutes eyes-closed), intelligence scores (Raven's Advanced Progressive Matrices scores, Raven and Court, 1998), Behavioral data (Age, Sex). 
Raw data of the main sample was aquired at Würzburg University, raw data of the replication sample was aquired at Frankfurt University.
The raw data analyzed during the current study are available from the corresponding author upon reasonable request.

## 4. Structure and Script description
### Main sample

For the analysis done in the paper, the scripts should be run in the following order:

1.	`get_complexity_main` - Preprocessing, computation of complexity measures (entropy and microstate measures)
                            Note that ICA components to remove need to be adapted manually (visual inspection) for artefact removal in preprocessing.
                            Preprocessing is mainly based on MNE (Gramfort et al., 2013) and Pyprep (Bigdely-Shamlo et al., 2015). 
  
2.	`analysis_main` - Factor analysis, single associations between complexity measures and intelligence, multimodal model to predict intelligence in main sample
  

### Replication sample

3.	`get_complexity_repli` - Preprocessing, computation of complexity measures (entropy and microstate measures)
                             Note that ICA components to remove need to be adapted manually (visual inspection) for artefact removal in preprocessing.   
  
4.	`analysis_repli` - Factor analysis, single associations between complexity measures and intelligence, multimodal model to predict intelligence in main sample

### Visualization

5. `plot-results` - Visualization of association between complexity measures and intelligence

### Additional analysis

6. `compare_mse` - Comparing patterns of association between multiscale entropy and intelligence in main and replication sample

### Functions

``

## 5. Software requirements
-	Python 3.8
-	

## References
1.	Raven JC, Court JH (1998) Manual for Raven’s progressive matrices and vocabulary scales.
2.	Gramfort A, Luessi M, Larson E, Engemann DA, Strohmeier D, Brodbeck C, Goj R, Jas M, Brooks T, Parkkonen L, Hämäläinen M (2013) MEG and EEG data analysis with MNE-Python. Front Neurosci 0:267.
3.	Bigdely-Shamlo N, Mullen T, Kothe C, Su K-M, Robbins KA (2015) The PREP pipeline: standardized preprocessing for large-scale EEG analysis. Front Neuroinform 9.
## Copyright
Copyright (cc) 2022 by Jonas Thiele

<a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-nd/4.0/88x31.png" /></a><br /><span xmlns:dct="http://purl.org/dc/terms/" property="dct:title">Files of BrainComplexity_Intelligence</span> by <a xmlns:cc="http://creativecommons.org/ns#" href="https://github.com/jonasAthiele/BrainReconfiguration_Intelligence" property="cc:attributionName" rel="cc:attributionURL">Jonas A. Thiele</a> are licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/">Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License</a>.

Note that external functions used may have other licenses.
