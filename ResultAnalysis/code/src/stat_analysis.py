from scipy.stats import spearmanr
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def make_scaler(data):
     scaler = MinMaxScaler(feature_range = (0, 1))
     scaler.fit(data)

     return scaler

def spearman_analysis (sbj_idx, y, distance, diameter, file_path = None, is_save = False):

    sbj = sbj_idx[0][0]

    corr1, p1 = spearmanr(distance, y, axis = 0)
    corr2, p2 = spearmanr(diameter, y, axis = 0) 
    corr3, p3 = spearmanr(distance, diameter, axis = 0)

    data = pd.DataFrame({'DPI_depth':      [round(corr1, 2), round(p1, 2)],
                         'diameter_depth': [round(corr2, 2), round(p2, 2)],
                         'DPI_diameter':   [round(corr3, 2), round(p3, 2)]},
                          index = ['corr', 'p_value'])
        
    if is_save:
        file_path.mkdir(parents=True, exist_ok=True)
        filename = file_path/f"sbj{sbj}_spearman.txt"
        data.to_csv(filename, sep = '\t')
