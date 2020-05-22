# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   Author :       Dechang Wang
   date：          2019/12/24
-------------------------------------------------
"""
import numpy as np
import pandas as pd
from sklearn import metrics
from numba import njit


def calc_information(train_score):
    data = np.around(train_score, 2)
    m, n = data.shape
    amn_matrix = calc_amn(data)
    c_matrix = np.zeros([m, n])
    for i in range(m):
        for j in range(n):
            c = 0
            cur_value = data[i, j]
            for other in range(n):
                if cur_value == data[i, j]:
                    c += 1
            c_matrix[i, j] = c

    weight_arr = np.zeros(n)
    for weight_index in range(n):
        for i in range(m):
            cur_c_value = c_matrix[i, weight_index]
            cur_r_ij = amn_matrix[weight_index][data[i, weight_index]]
            cur_columns_sum = c_matrix[:, j].sum()
            for j in range(n):
                if j == weight_index:
                    continue
                other_r_ij = amn_matrix[j][data[i, j]]
                c_possible = cur_c_value / cur_columns_sum
                cur_r_possible = cur_r_ij / m
                other_r_possible = other_r_ij / m
                info = c_possible * np.log((c_possible) / (cur_r_possible * other_r_possible))
                weight_arr[weight_index] += info
    # weight_arr = weight_arr/m
    return weight_arr


def calc_amn(data):
    row, columns = data.shape
    df = pd.DataFrame()
    amn_matrix = []
    for i in range(columns):
        df['value'] = data[:, i]
        vc = df['value'].value_counts()
        value_dic = dict()
        for key_value in vc.iteritems():
            value_dic[key_value[0]] = key_value[1]
        amn_matrix.append(value_dic)

    return amn_matrix


def calc_weight(original_weights, arr_info_entropy):
    weight = np.asarray(original_weights) / np.asarray(arr_info_entropy)
    weight = weight / weight.sum()
    return weight


def calc_nmi(a, b):
    a = np.around(a, decimals=2)
    b = np.around(b, decimals=2)
    return metrics.normalized_mutual_info_score(a, b)

# @njit
def calc_avg_nmi(data):
    data = np.around(data, decimals=2)
    m, n = data.shape
    res_arr = np.zeros([n,])
    for i in range(n):
        for j in range(n):
            if i != j:
                res_arr[i] += calc_nmi(data[:,i],data[:,j])
        res_arr[i] = res_arr[i] / (n-1)
    return res_arr