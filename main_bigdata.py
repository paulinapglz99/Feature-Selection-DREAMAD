# -*- coding: utf-8 -*-
"""main.ipynb
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
#for MI and VI
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import KBinsDiscretizer
from scipy.stats import entropy
#For PCA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import MultiLabelBinarizer

import sys
data = pd.read_csv(sys.argv[1])


#Import own functions
from definitions import (
    completeness_filter,
    fs_varianced,
    fs_linear_corr,
    fs_mi_vi_matrix,
    fs_pca,
    voting_matrix,
    filter_dataframe,
    flooring_capping
)

#Target vars

target_vars = ['ADNC',
 'Braak',
 'Thal',
 'CERAD',
 'LATE', 'LEWY',
 'percent 6e10 positive area',
 'percent AT8 positive area',
 'percent NeuN positive area',
 'percent GFAP positive area',
 'percent aSyn positive area',
 'percent pTDP43 positive area',
 ]

#Flooring and capping
data = flooring_capping(data, 0.1, 0.90)

#Run completeness with a threshold of 80%
_, completeness_vars, _ = completeness_filter(data, 80.0)

#Get variables that crossed the variance threshold
_, _, variance_vars = fs_varianced(data, quartile=3)

#Get the vars that survided the linear correlation filter
_, linear_filter_vars, _, _, _, _, _ = fs_linear_corr(data, zscore_threshold=2.0)

#Get the vars that survived the VI and MI filters
_, _, information_vars, _, _, _ = fs_mi_vi_matrix(data, target_vars, n_bins=10, threshold_quantile=0.5)

#Get the variables that survived the filtering by PCA
_, _, _, top_variables_pca, _, _, _ = fs_pca(data,  n_components=None, n_top_variables=50)

#Create vote matrix
filters = {
    "completeness_vars": completeness_vars,
    "variance_vars": variance_vars,
    "linear_vars": linear_filter_vars,
    "non_linear_vars": information_vars,
    "pca_vars": top_variables_pca

}

vote_matrix, winners = voting_matrix(filters, min_votes=3)
print("\nWinner variables:", winners)
final_data, _ = filter_dataframe(df=data, winners=winners)

_, _, _, _, _, _, final_scaled_data = fs_pca(final_data,  n_components=None, n_top_variables=50)

#Save data
final_data.to_csv("data/final_data.csv", index=False)
final_scaled_data.to_csv("data/final_scaled_data.csv", index=False)
