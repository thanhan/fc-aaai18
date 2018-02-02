#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 21:35:56 2017

@author: thanhan
"""

import util
import data_pp
import pickle
import pystan
import numpy as np
import pandas as pd


def clean_body_labels(data):
    """
    clean the stance labels for body text
    """
    n = len(data)
    data.index = range(n)
    for i in range(n):
        if data.iloc[i].astance == 'ignoring':
            data.set_value(i, 'astance', 'observing')
            
        #if data.iloc[i].astance == '':
        #    data.set_value(i, 'astance', data.iloc[i].articleHeadlineStance)
            
    return data
    

def get_processed_data():
    import features
    (train_data, X_train, val_data, X_val, test_data, X_test) = features.get_data()
    all_data = pd.concat( [train_data, val_data, test_data], ignore_index = True)
    data = data_pp.process_data(all_data)
    clean_body_labels(data)
    
    train_data_pp = data[: len(train_data)]
    val_data_pp = data[len(train_data): len(train_data) + len(val_data)]
    test_data_pp = data[len(train_data) + len(val_data): ]
    
    return (train_data_pp, X_train, val_data_pp, X_val, test_data_pp, X_test)
    

#(train_data, X_train, val_data, X_val, test_data, X_test) = pickle.load( open('edata.pkl'))
#all_data = pd.concat( [train_data, val_data, test_data], ignore_index = True)
#data = data_pp.process_data(all_data)

#train_data_pp = data[: len(train_data)]
#val_data_pp = data[len(train_data): len(train_data) + len(val_data)]
#test_data_pp = data[len(train_data) + len(val_data): ]

#(train_data_pp, X_train, val_data_pp, X_val, test_data_pp, X_test) = \
#pickle.load( open('edata_pp.pkl') )

#stan_input = data_pp.make_stan_input(train_data_pp, X_train, val_data_pp, X_val)

#sm = pickle.load( open('fact_model.pkl') )
#fit = sm.sampling(data=stan_input)

sm = None

def compile_stan():
    global sm
    sm = pystan.StanModel(file = 'fact_model.stan')
    
def run_stan(stan_input):
    fit = sm.vb(stan_input)
    return fit
    
def run_fact_model(n_em = 5):
    
    
    for em_it in range(n_em):
        # E-step
        fit = run_stan(stan_input)
        
        # M-step
        