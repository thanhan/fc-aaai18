#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 15:50:45 2017

@author: atn
"""

import crowd
import models
import pickle
import numpy as np
import pandas as pd
import util
import scipy
import active
import sys


(train_data_pp, X_train, val_data_pp, X_val, test_data_pp, X_test) = \
pickle.load( open('edata_pp_020.pkl') )
#pickle.load( open('edata_pp.pkl') )

#m = models.model(train_data_pp, X_train, val_data_pp, X_val, seed = 1, \
#                 sample_its=2000, burn = 1000)
#m.init_model()

# 1 - 300
data1 = crowd.read_batch_data(train_data_pp, fn = 'mturk/Batch_2768981_batch_results.csv')


data2 = crowd.read_batch_data(train_data_pp, fn = 'mturk/Batch_2770710_batch_results.csv')

# 400-500
data3 = crowd.read_batch_data(train_data_pp, fn = 'mturk/Batch_2774612_batch_results.csv')

# 500 - 600
data4 = crowd.read_batch_data(train_data_pp, fn = 'mturk/Batch_2775217_batch_results.csv')


# 600 - 700
data5 = crowd.read_batch_data(train_data_pp, fn = 'mturk/Batch_2777423_batch_results.csv')

# 700 - 900
data6 = crowd.read_batch_data(train_data_pp, fn = 'mturk/Batch_2779944_batch_results.csv')

# 900 - 1000
data7 = crowd.read_batch_data(train_data_pp, fn = 'mturk/Batch_2781073_batch_results.csv')

# 1000- 1100
data8 = crowd.read_batch_data(train_data_pp, fn = 'mturk/Batch_2785350_batch_results.csv')

# map from raw crowd labels to standard labels
stance_dic = {'For': 'for', 'Probably For': 'observing', 'Probably Against': 'observing', \
              'Neutral': 'observing', 'Against': 'against', 'Observing': 'observing'}

#stance_dic = {'For': 'for', 'Probably For': 'for', 'Probably Against': 'against', \
#              'Neutral': 'observing', 'Against': 'against'}



data8.ans = map(lambda x: stance_dic[x], data8.ans)

# 1100 - 1200
data9 = crowd.read_batch_data(train_data_pp, fn = 'mturk/Batch_2785372_batch_results.csv')
data9.ans = map(lambda x: stance_dic[x], data9.ans)

# 1200 - 1400
data10 = crowd.read_batch_data(train_data_pp, fn = 'mturk/Batch_2785468_batch_results.csv')
data10.ans = map(lambda x: stance_dic[x], data10.ans)

data_tvt = pd.concat([train_data_pp, val_data_pp, test_data_pp], ignore_index = True)

# 1400 - 1600
data11 = crowd.read_batch_data(data_tvt, fn = 'mturk/Batch_2785525_batch_results.csv')
data11.ans = map(lambda x: stance_dic[x], data11.ans)


# 1600 - 1800
data12 = crowd.read_batch_data(data_tvt, fn = 'mturk/Batch_2785634_batch_results.csv')
data12.ans = map(lambda x: stance_dic[x], data12.ans)

# 1800 - 2100
data13 = crowd.read_batch_data(data_tvt, fn = 'mturk/Batch_2785664_batch_results.csv')
data13.ans = map(lambda x: stance_dic[x], data13.ans)


# 2100 - 2200
data14 = crowd.read_batch_data(data_tvt, fn = 'mturk/Batch_2793340_batch_results.csv')
data14.ans = map(lambda x: stance_dic[x], data14.ans)



# 2100 - 2200 with rationale
data15 = crowd.read_batch_data(data_tvt, fn = 'mturk/Batch_2793517_batch_results.csv')
data15.ans = map(lambda x: stance_dic[x], data15.ans)

# 2200 - 2300 with rationale
data16 = crowd.read_batch_data(data_tvt, fn = 'mturk/Batch_2793686_batch_results.csv')
data16.ans = map(lambda x: stance_dic[x], data16.ans)

# 2200 - 2300 with rationale
data16 = crowd.read_batch_data(data_tvt, fn = 'mturk/Batch_2793686_batch_results_new.csv')
data16.ans = map(lambda x: stance_dic[x], data16.ans)



# 2300 - 2400 with rationale
data17 = crowd.read_batch_data(data_tvt, fn = 'mturk/Batch_2793845_batch_results.csv')
data17.ans = map(lambda x: stance_dic[x], data17.ans)

# 2400 - end with rationale
data18 = crowd.read_batch_data(data_tvt, fn = 'mturk/Batch_2794184_batch_results.csv')
data18.ans = map(lambda x: stance_dic[x], data18.ans)



# concat to get all crowd data
data = pd.concat((data1, data2, data3, data4, data5, data6, data7,\
                  data8, data9, data10, data11, data12, data13, \
                  data15, data16, data17, data18))

train_val_data_pp = pd.concat([train_data_pp, val_data_pp])
X_train_val = scipy.sparse.vstack((X_train, X_val))


erange = []
#erange = range(2072, 3000)
(data_all, X, cds, cdv, cds_test) = models.prepare_cm_data(train_val_data_pp, \
        X_train_val, test_data_pp, X_test, data, expert_range = erange, \
        train_range = 2071, test_range = 2596)

#cm = models.crowd_model(m.data_all, m.X, cds, cdv)
#cm.init_model()


def online_experiment():
    e = active.experiment(train_val_data_pp, X_train_val, \
                           test_data_pp, X_test, cds, cdv, cds_test)
    
    #e.do_exp(1, 1, 1, 1)
    #e.do_exp(2, 1, 1, 1)
    
    #e.do_exp(1, 1, 0, 0)
    #e.do_exp(2, 1, 0, 0)
    
    #e.do_exp(1, 0, 1, 0)
    #e.do_exp(2, 0, 1, 0)
    
    
    #e.do_exp(1, 0, 0, 1)
    
    #e.do_exp(2, 0, 0, 1)
    
    e.do_exp(1, 10, 1, 0)
    e.do_exp(2, 10, 1, 0)
    #e.do_exp(3, 10, 1, 0)
    pickle.dump(e.res, open('active_cmv3.pkl', 'w'))
    
    #e.do_exp(1, 10, 0, 1)
    #e.do_exp(2, 10, 0, 1)
    
    #e.do_exp(1, 10, 1, 1)
    #e.do_exp(2, 10, 1, 1)
    
    #e.do_exp(1, 1, 10, 0)
    #e.do_exp(2, 1, 10, 0)
    
    #e.do_exp(1, 1, 10, 1)
    #e.do_exp(2, 1, 10, 1)
    
    #e.do_exp(1, 0, 0, 0)
    #e.do_exp(2, 0, 0, 0)
    


def offline_experiment():
    #arg = int(sys.argv[1])
    # number of expert labels
    num = [1, 400, 800, 1200, 1600, 2000]
    res_bl  = [[],[],[],[],[],[]]
    res_cm  = [[],[],[],[],[],[]]
    res_cmv = [[],[],[],[],[],[]]
    save_cm = []

    #for seed in [arg]:
    for seed in range(10, 20, 1):
        print "seed", seed
        rs = np.random.RandomState(seed = seed)
        for i in range(6):
            e = num[i]
            erange = rs.permutation(2071)[:e]
            #erange = range(2071)
            (data_all, X, cds, cdv, cds_test) = models.prepare_cm_data(train_val_data_pp, \
                X_train_val, test_data_pp, X_test, data, expert_range = erange, \
                train_range = 2071, test_range = 2595)
        
            cmv = models.model_cv(data_all, X, cds, cdv)
            cmv.init_model()
            
            cm = models.crowd_model(data_all, X, cds, cdv)
            cm.init_model()
            
            
            #(ps, pt) = util.baseline_crowd(train_val_data_pp, X_train_val,\
            #    test_data_pp, X_test, cds, cdv)
            
            (ps, pt, pp_s, pp_t, clf_st, clf_vera) = util.baseline_crowd(\
            train_val_data_pp, X_train_val, test_data_pp, X_test, cds, cdv, return_proba=True)
            
            print e
            print util.get_acc(test_data_pp, ps, pt, pp_t)
            res_bl[i].append(util.get_acc(test_data_pp, ps, pt, pp_t))
            
            cm.em(3)
            cmv.em(3)
            
            save_cm.append(cm)
            
            res_cm[i].append(util.get_acc(test_data_pp, cm.res_s[2071:], \
                  cm.res_v[240:], cm.pos_v[240:]))
            
            res_cmv[i].append(util.get_acc(test_data_pp, cmv.res_s[2071:], \
                   cmv.res_v[240:], cmv.alpha[240:]))
            
            #print util.get_acc(test_data_pp, cm.res_s[2071:], cm.res_v[240:])
            print util.get_acc(test_data_pp, cmv.res_s[2071:], \
                   cmv.res_v[240:], cmv.alpha[240:])
            print '------------------'
            
            
    res_bl = np.asarray(res_bl)
    print np.mean(res_bl, 1)
    
    res_cmv = np.asarray(res_cmv)
    print np.mean(res_cmv, 1)
    
    res_cm = np.asarray(res_cm)
    print np.mean(res_cm, 1)
    
    pickle.dump([res_bl, res_cmv, res_cm], open('res_all.pkl', 'w'))

if __name__ == "__main__":
    if sys.argv[1] == 'online':
        print 'online experiment'
        online_experiment()
    elif sys.argv[1] == 'offline':
        print 'offline experiment'
        offline_experiment()
