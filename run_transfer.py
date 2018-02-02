#from run_crowd import *
#import models

#erange = range(2071)
#erange = range(2071)
#(data_all, X, cds, cdv, cds_test) = models.prepare_cm_data(train_val_data_pp, \
#    X_train_val, test_data_pp, X_test, data, expert_range = erange, \
#    train_range = 2071, test_range = 2595)

#cmv = models.model_cv(data_all, X, cds, cdv)
#cmv.init_model()

import pickle
cmv_clf_stance = pickle.load(open('clf/cmv_clf_stance.pkl') )
bl_clf_stance  = pickle.load(open('clf/baseline_clf_st.pkl'))

import util


a = util.snopes_read(loc='')

a = util.snopes_pp(a)

(a_train, a_val, a_test) = util.snopes_split(a)


import app
f_train = app.get_features(a_train)
f_val   = app.get_features(a_val)
f_test   = app.get_features(a_test)


import pandas as pd
import scipy
# set train data = train + val
a_train = pd.concat((a_train, a_val))
a_train.index = range(len(a_train))
f_train = scipy.sparse.vstack((f_train, f_val))

import numpy as np
import data_pp

def number_articles(train, test):
    """
    number articleCount from 1, test continues from train
    number claims from 1, test continues from train
    re-index
    """
    train.articleCount = 1 + np.arange(len(train))
    test.articleCount  = 1 + np.arange(len(train), len(train) + len(test), 1)
    
    train.index = range(len(train))
    test.index  = range(len(test))
    
    
    claims = pd.Series(train.claimCount).unique() 
    dic_claims = {c: (i+1) for i, c in enumerate(claims)}
    train = train.assign(claimCount = data_pp.apply_dic(train, dic_claims, 'claimCount'))
    
    train_claims = len(claims)
    claims = pd.Series(test.claimCount).unique() 
    dic_claims = {c: (i+1+train_claims) for i, c in enumerate(claims)}
    test = test.assign(claimCount = data_pp.apply_dic(test, dic_claims, 'claimCount'))
    
    return (train, test)
    

def baseline(d_train, x_train, d_test, x_test):
    st = bl_clf_stance.predict(x_train)
    d_train['articleHeadlineStance'] = st
    d_test['articleHeadlineStance'] ='none'
    
    x = util.baseline(d_train, x_train, d_test, x_test, clf_stance=bl_clf_stance, \
                      source_len=1000, return_proba=True, pp_truth_permu=False)
    
    
    return util.get_acc(d_test, x[0], x[1], x[3], bin_brier=True)

import models

def new(a_train, f_train, a_test, f_test):
    """
    transfer using CMV (variational)
    """
    st2 = cmv_clf_stance.predict(f_train)
    a_train['articleHeadlineStance'] = st2
    a_test['articleHeadlineStance'] = cmv_clf_stance.predict(f_test)
    
    a_all = pd.concat([a_train, a_test], ignore_index=True)
    f_all = scipy.sparse.vstack((f_train, f_test))
        
    cf = models.model_transfer(a_all, f_all, [], [], n_train=len(a_train), vera_range=[0,1])
    
    cf.init_model(cmv_clf_stance)
    
    cf.em(3)
    
    return util.get_acc(a_test, cf.res_s[cf.n_train:], \
                   cf.res_v[cf.train_m:], cf.alpha[cf.train_m:], bin_brier=True)
    
    
res_bl  = [[],[],[],[],[],[],[],[]]
res_ne  = [[],[],[],[],[],[],[],[]]
num     = [100, 600, 1100, 1600, 2100, 2600, 3100, 3600]    
n = 8

for seed in range(10):
    rs = np.random.RandomState(seed = seed)
    for i, c in enumerate(num):
        #randomly select c claims
        c_subset = rs.permutation(3600)[:c]
        
        c_mask = a_train.apply(lambda x: x['claimCount'] in c_subset , axis=1)
        d_train = a_train[c_mask]
        x_train = scipy.sparse.csr_matrix(f_train.todense()[c_mask])
        x_test = f_test
        
        (d_train, d_test) = number_articles(d_train, a_test)
        
        
        r_bl = baseline(d_train, x_train, d_test, x_test)
        r_ne = new(d_train, x_train, d_test, x_test)
        
        print r_bl, r_ne
        
        res_bl[i].append(r_bl[2])
        res_ne[i].append(r_ne[2])
        
# plot
data = []
for i, c in enumerate(res_bl):
    for j, x in enumerate(c): data.append([j, num[i], 'Baseline', x])
for i, c in enumerate(res_ne):
    for j, x in enumerate(c): data.append([j, num[i], 'Ours', x])
data = pd.DataFrame(data, columns=['unit', 'time', 'condition', 'value'])

import seaborn as sns
g = sns.tsplot(data=data, unit='unit', time='time', condition='condition', value='value')
g.set(xlim=(0, 4000))

import active
active.plot_curves([res_bl.T, res_ne.T], time=range(100, 3601, 500), \
                   xlab='Training claims', ylab='Brier score', xl=3600, \
                   save_name='fig/snopes.pdf')

# plot old figs without gibbs
res = pickle.load(open('save_a2.pkl'))
res_bl = pickle.load(open('save_res_bl.pkl') )
active.plot_curves([res[0][:, :, 0].T, res[1][:, :, 0].T], ['Baseline', 'Our'], time = range(0, 2001, 400), xlab = 'Stance labels by journalists', ylab = 'Stance accuracy', save_name='fig/new_fact_stance.pdf')
active.plot_curves([res_bl[:, :, 2].T, res[1][:, :, 2].T], ['Baseline', 'Our'], time = range(0, 2001, 400), xlab = 'Stance labels by journalists', ylab = 'Brier score', save_name='fig/new_fact_brier.pdf')

eres1 = pickle.load(open('save_e3.pkl'))
eres1.update(pickle.load(open('save_e4.pkl')))
active.plot_curves([active.take_res(eres1[(1,10,1,0)], 0), active.take_res(eres1[(2,10,1,0)], 0)], ['Baseline', 'Ours'], time = range(0, 500, 50), xlab = 'Test-set stance labels by crowds', ylab = 'Stance accuracy', save_name='fig/new_active_stance.pdf')
active.plot_curves([active.take_res(eres1[(1,10,1,0)], 2), active.take_res(eres1[(2,10,1,0)], 2)], ['Baseline', 'Ours'], time = range(0, 500, 50), xlab = 'Test-set stance labels by crowds', ylab = 'Brier score', save_name='fig/new_active_brier.pdf')
