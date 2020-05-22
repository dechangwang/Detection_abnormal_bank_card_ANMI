import datetime
import time

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

from models.lof import LOF
from models.feature_bagging import FeatureBagging
from models.combination import aom, moa
from utils.stat_models import pearsonr
from utils.utility import get_local_region
from utils.utility import get_competent_detectors
from utils.utility import train_predict_lof, generate_bagging_indices
from utils.utility import print_save_result, save_script
from utils.utility import loaddata, precision_n_score, standardizer
from utils.entropy import calc_entropy
import os
import threading
import pandas as pd
from utils.avg_information import calc_information,calc_weight,calc_nmi,calc_avg_nmi


# access the timestamp for logging purpose
today = datetime.datetime.now()
timestamp = today.strftime("%Y%m%d_%H%M%S")

# set numpy parameters
np.set_printoptions(suppress=True, precision=4)

###############################################################################
# parameter settings

#data = 'musk'
# data = 'letter'


base_detector = 'lof'
n_ite = 10  # number of iterations
test_size = 0.4  # training = 60%, testing = 40%
n_baselines = 20  # the number of baseline algorithms, DO NOT CHANGE



def train_and_predict(data_name):
    # reference pearson size:
    # https://www.researchgate.net/post/What_is_the_minimum_sample_size_to_run_Pearsons_R
    loc_region_size = 0
    loc_region_min = 30  # min local region size
    loc_region_max = 100  # max local region size
    ###############################################################################
    # adjustable parameters
    loc_region_perc = 0.1
    loc_region_ite = 20  # the number of iterations in defining local region
    loc_region_threshold = int(loc_region_ite / 2)  # the threshold to keep a point
    loc_min_features = 0.5  # the lower bound of the number of features to use

    n_bins = 10
    n_selected = 1  # actually not a parameter to tweak

    n_clf = 50
    k_min = 5
    k_max = 200

    # for SG_AOM and SG_MOA, choose the right number of buckets
    n_buckets = 5
    n_clf_bucket = int(n_clf / n_buckets)
    assert (n_clf % n_buckets == 0)  # in case wrong number of buckets

    # flag for printing and output saving
    verbose = True

    # record of feature bagging detector
    fb_n_neighbors = []
    ###############################################################################

    start_time = time.time()
   #data_name = 'pageblocks'
    xy_data = pd.read_csv(data_name)
    X_orig, y_orig = xy_data.iloc[:,:-1],xy_data.iloc[:,-1]

    # initialize the matrix for storing scores
    roc_mat = np.zeros([n_ite, n_baselines])  # receiver operating curve
    ap_mat = np.zeros([n_ite, n_baselines])  # average precision

    for t in range(n_ite):
        print('\nn_ite', t + 1, data_name )  # print status

        random_state = np.random.RandomState()

        # split the data into training and testing
        X_train, X_test, y_train, y_test = train_test_split(X_orig, y_orig,
                                                            test_size=test_size,
                                                            random_state=random_state)
        # in case of small datasets
        if k_max > X_train.shape[0]:
            k_max = X_train.shape[0]
        k_list = random_state.randint(k_min, k_max, size=n_clf).tolist()
        k_list.sort()

        # normalized the data
        X_train_norm, X_test_norm = standardizer(X_train, X_test)

        train_scores = np.zeros([X_train.shape[0], n_clf])
        test_scores = np.zeros([X_test.shape[0], n_clf])

        # initialized the list to store the results
        test_target_list = []
        method_list = []

        # generate a pool of detectors and predict on test instances
        train_scores, test_scores = train_predict_lof(k_list, X_train_norm,
                                                      X_test_norm,
                                                      train_scores,
                                                      test_scores)

        #######################################################################
        # fit feature bagging using median of k_list
        # n_neighbors = int(np.median(k_list))
        n_neighbors = random_state.randint(low=k_min, high=k_max)
        clf = FeatureBagging(base_estimator=LOF(n_neighbors=n_neighbors),
                             n_estimators=len(k_list), check_estimator=False)
        print(clf)
        fb_n_neighbors.append(n_neighbors)
        clf.fit(X_train_norm)

        #######################################################################
        # generate normalized scores
        train_scores_norm, test_scores_norm = standardizer(train_scores,
                                                           test_scores)
        # generate mean and max outputs
        # SG_A and SG_M
        target_test_mean = np.mean(test_scores_norm, axis=1)
        target_test_max = np.max(test_scores_norm, axis=1)
        test_target_list.extend([target_test_mean, target_test_max])
        method_list.extend(['GG_a', 'GG_m'])

        # calculate every detector information entropy
        info_ent_arry = calc_avg_nmi(train_scores)
        # generate information entropy weighted mean
        # 使用信息熵加权计算
        target_test_entropy_weighted_pear = np.mean(
            test_scores_norm * info_ent_arry.reshape(1, -1), axis=1)

        test_target_list.append(target_test_entropy_weighted_pear)
        method_list.append('GG_a_ent')

        # generate weighted mean
        target_test_weighted_pear_max_ent = np.max(test_scores_norm*info_ent_arry.reshape(1, -1), axis=1)

        test_target_list.append(target_test_weighted_pear_max_ent)
        method_list.append('GG_m_ent')



        # generate pseudo target for training -> for calculating weights
        target_mean = np.mean(train_scores_norm, axis=1).reshape(-1, 1)
        target_max = np.max(train_scores_norm, axis=1).reshape(-1, 1)


        # 使用信息熵加权后的pseudo
        target_ent_mean = np.mean(train_scores_norm * info_ent_arry.reshape(1, -1) ,
                                  axis=1).reshape(-1, 1)
        target_ent_max = np.max(train_scores_norm * info_ent_arry.reshape(1, -1) ,
                                  axis=1).reshape(-1, 1)
        


        # generate average of maximum (SG_AOM) and maximum of average (SG_MOA)
        target_test_aom = aom(test_scores_norm, n_buckets, n_clf)

        target_test_aom_ent = aom(test_scores_norm * info_ent_arry.reshape(1, -1) , n_buckets, n_clf)

        test_target_list.extend([target_test_aom, target_test_aom_ent])
        method_list.extend(['GG_aom', 'GG_aom_ent'])
        ##################################################################

        # define the local region size
        loc_region_size = int(X_train_norm.shape[0] * loc_region_perc)
        if loc_region_size < loc_region_min:
            loc_region_size = loc_region_min
        if loc_region_size > loc_region_max:
            loc_region_size = loc_region_max

        # define local region
        ind_arr = get_local_region(X_train_norm, X_test_norm,
                                   loc_region_size,
                                   loc_region_ite=loc_region_ite,
                                   local_region_strength=loc_region_threshold,
                                   loc_min_features=loc_min_features,
                                   random_state=random_state)

        pred_scores_best = np.zeros([X_test.shape[0], ])
        pred_scores_ens = np.zeros([X_test.shape[0], ])
        pred_scores_best_ent = np.zeros([X_test.shape[0], ])
        pred_scores_ens_ent = np.zeros([X_test.shape[0], ])

        for i in range(X_test.shape[0]):  # iterate all test instance

            ind_k = ind_arr[i]

            # get the pseudo target: mean
            target_k = target_mean[ind_k,].ravel()

            target_ent_k = target_ent_mean[ind_k,].ravel()

            # get the current scores from all clf
            curr_train_k = train_scores_norm[ind_k, :]

            # initialize containers for correlation
            corr_pear_n = np.zeros([n_clf, ])
            corr_pear_n_ent = np.zeros([n_clf, ])

            for d in range(n_clf):
                corr_pear_n[d,] = pearsonr(target_k, curr_train_k[:, d])
                corr_pear_n_ent[d,] = pearsonr(target_ent_k,curr_train_k[:,d])#*info_ent_arry[d]
                #corr_pear_n_ent[d,] = calc_nmi(target_k,curr_train_k[:,d])
            # pick the best one
            best_clf_ind = np.nanargmax(corr_pear_n)
            pred_scores_best[i,] = test_scores_norm[i, best_clf_ind]

            best_clf_ind_ent = np.nanargmax(corr_pear_n_ent)
            pred_scores_best_ent[i,] = test_scores_norm[i, best_clf_ind_ent]


        test_target_list.extend([pred_scores_best,
                                 pred_scores_best_ent])
        method_list.extend(['LSCP_a','LSCP_a_ent'])
        ######################################################################

        pred_scores_best = np.zeros([X_test.shape[0], ])
        pred_scores_ens = np.zeros([X_test.shape[0], ])

        pred_scores_best_ent = np.zeros([X_test.shape[0], ])
        pred_scores_ens_ent = np.zeros([X_test.shape[0], ])

        for i in range(X_test.shape[0]):  # iterate all test instance
            # get the neighbor idx of the current point
            ind_k = ind_arr[i]
            # get the pseudo target: mean
            target_k = target_max[ind_k,].ravel()
            target_k_ent = target_ent_max[ind_k,].ravel()

            # get the current scores from all clf
            curr_train_k = train_scores_norm[ind_k, :]

            # initialize containers for correlation
            corr_pear_n = np.zeros([n_clf, ])
            corr_pear_n_ent = np.zeros([n_clf, ])

            for d in range(n_clf):
                corr_pear_n[d,] = pearsonr(target_k, curr_train_k[:, d])
                corr_pear_n_ent[d,] = pearsonr(target_k_ent,curr_train_k[:, d])#*info_ent_arry[d]
                #corr_pear_n_ent[d,] = calc_nmi(target_k,curr_train_k[:,d])
            # pick the best one
            best_clf_ind = np.nanargmax(corr_pear_n)
            pred_scores_best[i,] = test_scores_norm[i, best_clf_ind]

            pred_scores_ens[i,] = np.mean(
                test_scores_norm[
                    i, get_competent_detectors(corr_pear_n, n_bins,
                                               n_selected)])

            # 使用信息熵加权后的pseudo
            best_clf_ind_ent = np.nanargmax(corr_pear_n_ent)
            pred_scores_best_ent[i,] = test_scores_norm[i, best_clf_ind_ent]

            pred_scores_ens_ent[i,] = np.mean(
                test_scores_norm[
                    i, get_competent_detectors(corr_pear_n_ent, n_bins,
                                               n_selected)])

        test_target_list.extend([pred_scores_best,pred_scores_best_ent,
                                 pred_scores_ens,pred_scores_ens_ent])
        method_list.extend(['LSCP_m','LSCP_m_ent',
                            'LSCP_aom','LSCP_aom_ent'])

        ######################################################################

        # store performance information and print result
        for i in range(n_baselines):
            roc_mat[t, i] = roc_auc_score(y_test, test_target_list[i])
            ap_mat[t, i] = average_precision_score(y_test,
                                                   test_target_list[i])
            print(method_list[i], roc_mat[t, i])
        print('local region size:', loc_region_size)

    print("--- %s seconds ---" % (time.time() - start_time))
    execution_time = time.time() - start_time

    # save parameters
    save_script(data_name, base_detector, timestamp, n_ite, test_size, n_baselines,
                loc_region_perc, loc_region_ite, loc_region_threshold,
                loc_min_features, loc_region_size, loc_region_min,
                loc_region_max, n_clf, k_min, k_max, n_bins, n_selected,
                n_buckets, fb_n_neighbors, execution_time,res_path='results_avg_nmi')

    # print and save the result
    # default location is /results/***.csv
    print_save_result(data_name, base_detector, n_baselines, roc_mat,
                      ap_mat, method_list, timestamp, verbose,res_path='results_avg_nmi')



if __name__ == '__main__':
    
    columns = os.listdir('csv_datasets')
    
    for data_name in columns:
        try:
            train_and_predict(data_name)
        except:
            print(data_name)
