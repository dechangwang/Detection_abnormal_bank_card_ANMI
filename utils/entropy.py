# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   Author :       Dechang Wang
   date：          2019/12/19
-------------------------------------------------
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def calc_entropy(data,standard=False):
    row, columns = data.shape
    info_entropy_arr = []
    data = np.around(data, 2)
    df = pd.DataFrame()
    for i in range(columns):

        df['value'] = data[:, i]
        vc = df['value'].value_counts()

        value_dic = dict()
        total_value_num = row
        information_entropy = 0
        for key_value in vc.iteritems():
            value_dic[key_value[0]] = key_value[1]

            # calucate entropy
            probability = key_value[1] / total_value_num
            information_entropy += probability * np.log(probability)

        info_entropy_arr.append(-1 * information_entropy)
    if standard:
        return StandardScaler().fit_transform(np.asarray(info_entropy_arr).reshape(-1,1)).ravel()
    return np.asarray(info_entropy_arr)
