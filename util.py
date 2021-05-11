# -*- coding: utf-8 -*-

import csv
#from matplotlib import pyplot as plt
#import matplotlib
import numpy as np
import sklearn
import crowd
import pandas as pd
import urlparse
import re
import tld
import os
import json
import data_pp

import models


#find an attribute from aid
def find_from_aid(data_raw, aid, a):
    for d in data_raw:
        if d['articleId'] == aid:
            if a in d and d[a] != "":
                return d[a]
    return ""


def input():
    f = open('emergent/url-versions-2015-06-14.csv')
    reader = csv.DictReader(f)
    data_raw = list(reader)
    data_raw_aid = [x['articleId'] for x in data_raw]

    f = open('emergent/url-versions-2015-06-14-clean.csv')
    reader = csv.DictReader(f)
    data_clean = list(reader)
    data_clean_aid = [x['articleId'] for x in data_clean]

#s = [(i, data_raw[i]['claimHeadline'], data_raw[i]['articleHeadlineStance'], data_raw[i]['articleStance']) \
     #for i, x in enumerate(data_aid) if x not in data_clean_aid]

#a = [d for d in data if d['articleHeadlineStance'] != d['articleStance'] ]
    data = []
    for dc in data_clean:
        d = dict(dc)
        aid = d['articleId']
        for x in ['articleStance', 'articleBody', 'claimTruthiness']:
            d[x] = find_from_aid(data_raw, aid, x)
        
        data.append(d)
    return data


# return dic from article ID to url/ body/ stance
def get_dic_aid():
    f = open('emergent/url-versions-2015-06-14.csv')
    reader = csv.DictReader(f)
    data_raw = list(reader)
    dic_url  = {d['articleId'] : d['articleUrl'] for d in data_raw}
    dic_body = {d['articleId'] : d['articleBody'] for d in data_raw}
    dic_as = {d['articleId'] : d['articleStance'] for d in data_raw} # article stance
    
    return (dic_url, dic_body, dic_as)
    
# return dic from claimId to truth label
def get_dic_truth():
    data = input()
    res = {d['claimId'] : d['claimTruthiness'] for d in data}
    return res

#dic of claim id to documents about that claim
def get_dic_claims(data):
    dic_claims = {}
    for d in data:
        cid = d['claimId']
        if cid not in dic_claims:
            dic_claims[cid] = []
        dic_claims[cid].append(d)
    return dic_claims

# 175 claims have ground truth
def count_claim(docs, stance = 'for'):
    res = 0
    for doc in docs:
        if doc['articleHeadlineStance'] == stance:
           res += 1
    return res

def plot_claims(dic_claims):
    for c, docs in dic_claims.items():
        if True:
            label = docs[0]['claimTruthiness']
            claims_for = count_claim(docs, 'for') + np.random.normal()/10
            claims_against = count_claim(docs, 'against')  + np.random.normal()/10
            claims_observing = count_claim(docs, 'observing')
            if label == 'true':
                h_true = plt.scatter(claims_for, claims_against, marker = 'P', color = 'green', alpha = 0.5)
            elif label == 'false':
                h_false = plt.scatter(claims_for, claims_against, marker = 'X', color = 'red', alpha = 0.5)            
            elif label == 'unknown':
                h_u = plt.scatter(claims_for, claims_against, marker = 'o', color = 'yellow', alpha = 0.5)         
    plt.xlabel('Articles for')
    plt.ylabel('Articles against')
    plt.legend([h_true, h_false, h_u], ['True', 'False', 'Unknown'])
    plt.savefig('claims.png', dpi = 600)
            #print predict, label, claims_for, claims_against, claims_observing
            
        
def get_dic_claim_acount(data):
    d = {}
    for i in range(len(data)):
        c = data.iloc[i]['claimCount']
        a = data.iloc[i]['articleCount']
        if c not in d: d[c] = 0
        d[c] += 1
        
    return d
        
def read_stan_csv(file_loc):
    f = open(file_loc)
    r = csv.reader(f)
    l = list(r)    
    names = l[4]
    n = len(l) - 7
    m = len(names)
    a = np.zeros( shape = (n, m))
    
    for i in range(7, len(l)):
        a[i-7, :] = map(float, l[i])
    
    f.close()
    return (names, a)
        

#'/tmp/tmpIRAlV_/output.csv'

def eval_stan1(names, a, test_data, X_test):
    i = names.index('ws.1')
    w = a[:, i: i+518]
    
    b = w.dot(X_test.toarray().T)
    c = np.mean(b, 0)
    
    l = np.mean(a[:, names.index('c.1')])
    r = np.mean(a[:, names.index('c.2')])
    
    f = lambda x: 'against' if x < l else 'observing' if x < r else 'for'
    d = map(f, c)
    
    return d


def predict_stance(a):
    """
    a: array of (n, 3)
    """
    n = a.shape[0]
    res = []
    
    for i in range(n):
        x = np.argmax(a[i, :])
        if x == 0: 
            res.append('against')
        elif x == 1:
            res.append('observing')
        else:
            res.append('for')
    return res


def get_name_index(names, x, i, j):
    try:
        return names.index(x + '.' + str(i) + '.' + str(j))
    except Exception:
        return names.index(x + '[' + str(i-1) + ',' + str(j-1) + ']')
        

def eval_stan2(fit, names = None, a = None):
    if names == None:
        (names, a) = read_stan_csv(fit['args']['sample_file'])
    res_claim = []
    #start_idx = names.index('claim_score.1.1') + 180 - 1
    start_idx = get_name_index(names, 'claim_score', 1, 1) + 180 - 1
    for i in range(start_idx, start_idx + 60):
        x = np.argmax([np.mean(a[:, 1 + i]), np.mean(a[:, 241 + i]), np.mean(a[:, 481 + i])])
        #count = [0, 0, 0]
        #for pos in range(1000):
        #    x = np.argmax([np.mean(a[pos, 1 + i]), np.mean(a[pos, 241 + i]),\
        #                   np.mean(a[pos, 481 + i])])
        #    count[x] += 1
        #x = np.argmax(count)
        if x == 0: 
            res_claim.append('false')
        elif x == 1:
            res_claim.append('unknown')
        else:
            res_claim.append('true')
            
    stance_a = np.zeros((582, 3))
    start_idx1 = get_name_index(names, 'stance_score', 1, 1)
    start_idx2 = get_name_index(names, 'stance_score', 1, 2)
    start_idx3 = get_name_index(names, 'stance_score', 1, 3)
    for i in range(582):
        j1 = start_idx1 + 1489 + i
        j2 = start_idx2 + 1489 + i
        j3 = start_idx3 + 1489 + i
        x = [np.mean(a[:, j1]), np.mean(a[:, j2]), np.mean(a[:, j3])]
        stance_a[i, :] = x
    
    res_stance = predict_stance(stance_a)
    return (res_claim, res_stance)


def convert_stance(a):
    dic = {'against': 1, 'observing': 2, 'for': 3}
    f = lambda x: dic[x]
    return np.asarray(map(f, a))
    
    
def eval_mul_lr(names, a, X_test):
    # ws in 518 x 3
    ws = np.zeros(shape = (518, 3))
    i = names.index('ws.1.1')
    ws[:, 0] = np.mean(a[:, i:i+518], 0)
    
    i = names.index('ws.1.2')
    ws[:, 1] = np.mean(a[:, i:i+518], 0)
    
    i = names.index('ws.1.3')
    ws[:, 2] = np.mean(a[:, i:i+518], 0)
    
    i = names.index('stance_intercept.1')
    intercept = np.mean(a[:, i: i+3], 0)
    
    print intercept
    
    n = X_test.shape[0]
    res = []
    for i in range(n):
        x = np.argmax(intercept + X_test[i].dot(ws))
        if x == 0:
            res.append('against')
        elif x == 1:
            res.append('observing')
        else:
            res.append('for')
    return res


def eval_mul_lr2(names, a, X_test):
    # ws in 518 x 3
    ws = np.zeros(shape = (518*3,))
    i = names.index('ws.1.1')
    ws = np.mean(a[:, i:i+3*518], 0)
    ws = ws.reshape( (518, 3))
    ws = ws.T
    
    
    
    
    n = X_test.shape[0]
    res = np.zeros((n,))
    for i in range(n):
        res[i] = np.argmax(ws.dot(X_test.toarray()[i, :])) + 1
    return res


def get_stance_score(stance, truth):
    if stance == 'for' and truth == 'true': return 1
    if stance == 'against' and truth == 'false': return 1
    return 0

def extract_truth_labels(data):
    claims = sorted(data.claimCount.unique().tolist())
    l = [''] * len(claims)
    for i in range(len(data)):
        row = data.iloc[i]
        truth = row.claimTruth
        claim = row.claimCount
        claimIdx = claims.index(claim)
        l[claimIdx] = truth
        
    return (claims, l)

def get_indicator_features(data, stances, source_len = 724):
    """
    indicator features for claims
    """
    dic_f = {} # claimCount -> features
    
    for i in range(len(data)):
        row = data.iloc[i]
        stance = stances[i]
        stance_id = 0 if stance == 'against' else 1 if stance == 'observing'\
            else 2
        source = row.sourceCount - 1 # 1-index to 0-index
        claim = row.claimCount
        
        if claim not in dic_f: dic_f[claim] = np.zeros((3, source_len))
        dic_f[claim][stance_id][source] = 1
    
    claims = dic_f.keys()
    for c in claims: 
        dic_f[c] = dic_f[c].flatten()
        
    return dic_f

def get_features(data, stances, source_len = 724):
    """
    features for claims
    """
    dic_f = {} # claimCount -> features
    
    for i in range(len(data)):
        row = data.iloc[i]
        stance = stances[i]
        stance_id = -1 if stance == 'against' else 0 if stance == 'observing'\
            else 1
        source = row.sourceCount - 1 # 1-index to 0-index
        claim = row.claimCount
        
        if claim not in dic_f: dic_f[claim] = np.zeros((source_len,))
        dic_f[claim][source] = stance_id
    
    #claims = dic_f.keys()
    return dic_f

def baseline(train_data, X_train, test_data, X_test, return_proba = False, \
             test_pos = None, clf_stance=None, source_len = 724, pp_truth_permu=True):
    """
    train_data include DS labels
    some of test_data may include labels
    """
    
    # classify stances
    if clf_stance == None:
        clf_stance = sklearn.linear_model.LogisticRegression(penalty='l1')
        clf_stance.fit(X_train, train_data.articleHeadlineStance)
    
    # get features for train data
    dic_f = get_features(train_data, \
            train_data.articleHeadlineStance, source_len=source_len)
        
    (train_claims, train_y) = extract_truth_labels(train_data)
    n = len(train_claims)
    m = dic_f.items()[0][1].shape[0]
    
    train_f = np.zeros((n, m))
    for i, c in enumerate(train_claims): train_f[i, :] = dic_f[c]
    
    # claim classifier
    #clf_claim = sklearn.linear_model.LogisticRegression(multi_class='multinomial',\
    #                                                    solver = 'lbfgs')
    #clf_claim = sklearn.linear_model.LogisticRegression(penalty = 'l1')
    clf_claim = sklearn.linear_model.LogisticRegression()
    clf_claim.fit(train_f, train_y)
    
    
    # predict on test data
    predicted_stances = clf_stance.predict(X_test)
    for i in range(len(predicted_stances)):
        if test_data.iloc[i]['articleHeadlineStance'] != 'none':
            predicted_stances[i] = test_data.iloc[i]['articleHeadlineStance']
            
    
    pp_stances = clf_stance.predict_proba(X_test)

    #if test_pos != None: pp_stances = pp_stances * test_pos
    #predicted_stances = np.argmax(pp_stances, 1)
    #num2lab = {0: 'against', 1: 'observing', 2: 'for', -1: 'nan'}
    #predicted_stances = map(lambda x: num2lab[x], predicted_stances)
    
    # get features for test data
    dic_f_test = get_features(test_data, predicted_stances, source_len=source_len)
    
    test_claims = sorted(test_data.claimCount.unique().tolist())
    n = len(test_claims)
    m = dic_f_test.items()[0][1].shape[0]
    
    test_f = np.zeros((n, m))
    for i, c in enumerate(test_claims): test_f[i, :] = dic_f_test[c]
    
    # predict vera of test data
    predicted_truth = clf_claim.predict(test_f)
    
    if not return_proba:
        return (predicted_stances, predicted_truth)

    
    pp_truth   = clf_claim.predict_proba(test_f)
    
    if pp_truth_permu:
        pp_truth = pp_truth[:, [0, 2, 1] ]
    return (predicted_stances, predicted_truth, pp_stances, pp_truth, \
            clf_stance, clf_claim)


def vera_lab2num(a):
    m = {'false': -1, 'unknown': 0, 'true': 1}
    return [m[x] for x in a]

def mul_brier(y_true, y_pred):
    """
    assume y_true in (-1, 0, 1)
    """
    n = len(y_true)
    res = 0
    m = y_pred.shape[1]
    
    for i in range(n):
        for j in range(m):
            o = 1 if y_true[i] + 1 == j else 0 # -1 index to 0 index
            res += pow(y_pred[i][j] - o, 2)
    return res * 1.0 / n

def binary_brier(y_true, y_pred):
    """
    binary brier score
    """
    n = len(y_true)
    res= 0
    
    for i in range(n):
        for j in range(2):
            o = 1 if y_true[i] == j else 0
            res += pow(y_pred[i][j] - o, 2)
            
    return res * 1.0 / n

def get_acc(test_data, predicted_stances, predicted_truth, pp_t = None, \
            bin_brier=False):
    stance_acc = sklearn.metrics.accuracy_score(test_data.articleHeadlineStance, \
                                                predicted_stances)
    
    (test_claims, labels) = extract_truth_labels(test_data)
    claim_acc =  sklearn.metrics.accuracy_score(labels, \
                                                predicted_truth)
    
    #claim_mae = sklearn.metrics.mean_absolute_error(vera_lab2num(labels), \
    #                                                vera_lab2num(predicted_truth))
    
    #print sklearn.metrics.confusion_matrix(labels, predicted_truth)
    if pp_t is not None:
        #print 'll = ', sklearn.metrics.log_loss(vera_lab2num(labels), pp_t)
        #print 'bs = ', mul_brier(vera_lab2num(labels), pp_t)
        if bin_brier:
            # false -> 0, true -> 1
            y_true = [0 if x == 'false' else 1 for x in labels]            
                
            bs = binary_brier(y_true, pp_t)
        else:
            bs = mul_brier(vera_lab2num(labels), pp_t)
        
        return (stance_acc, claim_acc, bs)
        
    
    return (stance_acc, claim_acc)

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def stance_num2lab(a):
    dic_s = {0: 'against', 1: 'observing', 2: 'for'}
    return map(lambda x: dic_s[x], a)

def vera_num2lab(a):
    dic_v = {0: 'false', 1: 'unknown', 2: 'true'}        
    return map(lambda x: dic_v[x], a)


def baseline_crowd(train_data, X_train, test_data, X_test, cds, cdv, \
                   return_proba = False, ds_it = 10, use_body= False, body_acc = 0.7):
    """
    baseline using crowd labels
    use_body: use stances of body text
    """
    dss = crowd.DS(cds, list_expert=[cds.expert_wid])
    #dss.em(5)
    dss.em(ds_it)
    dss.get_full_pos(range(len(train_data) + len(test_data) + 1))
    
    train = train_data.copy()
    test = test_data.copy()
    
    
    for i in range(len(train)):
        aid = train.iloc[i]['articleCount']
        #train.iloc[i]['articleHeadlineStance'] = dss.dic_al[aid]
        j = train.index.values[i]
        train.set_value(j, 'articleHeadlineStance', dss.dic_al[aid])
        
    test_pos = []
    for i in range(len(test)):
        aid = test.iloc[i]['articleCount']
        j = test.index.values[i]
        test_pos.append(dss.full_pos[aid])
        if aid in dss.dic_al:
            test.set_value(j, 'articleHeadlineStance', dss.dic_al[aid])
        else:
            test.set_value(j, 'articleHeadlineStance', "none")
    
    #print len(test)
    if use_body:
        return baseline_with_body(train, X_train, test, X_test, return_proba, \
                                  test_pos, body_acc = body_acc)
    else:
        return baseline(train, X_train, test, X_test, return_proba, test_pos)



def get_dic_sources(data):
    """
    dic from source to id
    """
    res = {}
    for row in data.itertuples():
        source = row.source
        sid    = row.sourceCount
        res[source] = sid
    return res


def build_claim_df(data):
    """
    data: data frame of articles: data_tvt
    data frame for claims for output to csv
    include (headline ID) source : headline
    """
    
    # dic: claimCount -> paragraph of headlines
    dic_hl = {}
    #dic: claimCount -> claimHeadline
    dic_ch = {}
    for row in data.itertuples():
        cl     = row.claimHeadline
        cc     = row.claimCount
        hl     = row.articleHeadline
        source = row.source
        
        dic_ch[cc] = cl
        
        if cc not in dic_hl: dic_hl[cc] = ""
        st = dic_hl[cc]
        n = st.count('\n') / 2
        dic_hl[cc] = st + "(" + str(n+1) + ") " + source + ": " +\
                     hl + "<br/><br/>"
        
    n_claims = len(dic_hl)
    df = pd.DataFrame(index = range(n_claims), columns = ['claimCount', 'headlines', \
                      'claimH'] )
    
    hl = dic_hl.items()
    for i in range(n_claims):
        df.loc[i] = list(hl[i]) + [ dic_ch[hl[i][0]] ]
    return df
    
def build_claim_df2(data, dc):
    """
    data: data frame of articles
    dc:   data of crowd labels
    """
    
    # dic: aid -> List of crowd labels
    dic_ac = {}
    for r in dc.itertuples():
        if r.aid not in dic_ac: dic_ac[r.aid] = ""
        dic_ac[r.aid] += str(r.ans) + " "
        
    for aid in dic_ac:
        dic_ac[aid] = ' '.join(sorted(dic_ac[aid].split()))
    
    
    #dic: claimCount -> claimHeadline
    dic_ch = {}
    
    # dic: claimCount -> paragraph of source: crowd labels
    dic_cc = {}
    for r in data.itertuples():
        cl     = r.claimHeadline
        cc     = r.claimCount
        hl     = r.articleHeadline
        source = r.source
        crowd = dic_ac[r.articleCount]
        
        dic_ch[cc] = cl
        
        if cc not in dic_cc: dic_cc[cc] = ""
        st = dic_cc[cc]
        n = st.count('<br/>') / 2
        dic_cc[cc] = st + "(" + str(n+1) + ") " + source + ": " +\
                     crowd + "<br/><br/>"
    
    #return dic_cc
    
    n_claims = len(dic_cc)
    df = pd.DataFrame(index = range(n_claims), columns = ['claimCount', 'stances', \
                      'claimH'] )
    
    hl = dic_cc.items()
    for i in range(n_claims):
        df.loc[i] = list(hl[i]) + [ dic_ch[hl[i][0]] ]
    return df


# for NER:

def get_headlines(data):
    h = data[ ['articleHeadline', 'articleHeadlineStance']]
    
    
    for_h        = h[h.articleHeadlineStance == 'for'].articleHeadline
    against_h    = h[h.articleHeadlineStance == 'against'].articleHeadline
    observing_h  = h[h.articleHeadlineStance == 'observing'].articleHeadline
    
    return (h, for_h, against_h, observing_h)

    
def get_claims(data):
    d = data[['claimHeadline', 'claimTruth']]
    
    dd = d.drop_duplicates()
    
    true_claims = dd[dd.claimTruth == 'true'].claimHeadline
    false_claims = dd[dd.claimTruth == 'false'].claimHeadline
    unknown_claims = dd[dd.claimTruth == 'unknown'].claimHeadline
    
    return (dd, true_claims, false_claims, unknown_claims)


# for error analysis

def output_wrong_stances(test_data, predicted_stances):
    true_stances = test_data.articleHeadlineStance.tolist()
    n = len(predicted_stances)
    
    for i in range(n):
        if true_stances[i] != predicted_stances[i]:
            print "Claim: ", test_data.iloc[i].claimHeadline, "\nHL:", test_data.iloc[i].articleHeadline
            print 'True stance: ', true_stances[i], "  ", "Predicted stance: ", predicted_stances[i], "\n"

    


def noise_stances(s, a = 0.7, seed = 1):
    """
    put noise into a list s of stances
    return a list with accuracy a
    """
    rs = np.random.RandomState(seed = seed)
    
    labels = ['against', 'observing', 'for']
    
    res = []
    for x in s:
        if rs.rand() < a:
            res.append(x)
        else:
            i = rs.randint(3)
            if labels[i] != x:
                res.append(labels[i])
            else:
                res.append(labels[(i + 1) % 3])
            
    return res
    
    

def baseline_with_body(train_data, X_train, test_data, X_test, return_proba = False, \
             test_pos = None, body_acc = 0.7):
    """
    baselines model w. body stance as an additional features to predict veracity
    body_acc: simulated accuracy of body stances
    """
    
    # classify stances
    clf_stance = sklearn.linear_model.LogisticRegression(penalty='l1')
    clf_stance.fit(X_train, train_data.articleHeadlineStance)
    
    # get features for train data (to classify veracity)
    dic_f = get_features(train_data, train_data.articleHeadlineStance)
    
    # features from body text stance
    dic_f2 = get_features(train_data, noise_stances(train_data.astance, body_acc))
        
    (train_claims, train_y) = extract_truth_labels(train_data)
    n = len(train_claims)
    m = dic_f.items()[0][1].shape[0]
    m2 = dic_f2.items()[0][1].shape[0]
    
    train_f = np.zeros((n, m + m2))
    for i, c in enumerate(train_claims): 
        train_f[i, :m] = dic_f[c]
        train_f[i, m:] = dic_f2[c]
    
    # claim classifier
    #clf_claim = sklearn.linear_model.LogisticRegression(multi_class='multinomial',\
    #                                                    solver = 'lbfgs')
    #clf_claim = sklearn.linear_model.LogisticRegression(penalty = 'l1')
    clf_claim = sklearn.linear_model.LogisticRegression()
    clf_claim.fit(train_f, train_y)
    
    
    # predict stances on test data, use provided labels when available
    predicted_stances = clf_stance.predict(X_test)
    for i in range(len(predicted_stances)):
        if test_data.iloc[i]['articleHeadlineStance'] != 'none':
            predicted_stances[i] = test_data.iloc[i]['articleHeadlineStance']
            
    
    pp_stances = clf_stance.predict_proba(X_test)

    #if test_pos != None: pp_stances = pp_stances * test_pos
    #predicted_stances = np.argmax(pp_stances, 1)
    #num2lab = {0: 'against', 1: 'observing', 2: 'for', -1: 'nan'}
    #predicted_stances = map(lambda x: num2lab[x], predicted_stances)
    
    # get features for test data
    dic_f_test = get_features(test_data, predicted_stances)
    dic_f2_test = get_features(test_data, noise_stances(test_data.astance.tolist(), body_acc))
    
    test_claims = sorted(test_data.claimCount.unique().tolist())
    n = len(test_claims)
    m = dic_f_test.items()[0][1].shape[0]
    m2 = dic_f2_test.items()[0][1].shape[0]
    
    test_f = np.zeros((n, m + m2))
    for i, c in enumerate(test_claims): 
        test_f[i, :m] = dic_f_test[c]
        test_f[i, m:] = dic_f2_test[c]
    
    # predict vera of test data
    predicted_truth = clf_claim.predict(test_f)
    
    if not return_proba:
        return (predicted_stances, predicted_truth)

    
    pp_truth   = clf_claim.predict_proba(test_f)
    pp_truth = pp_truth[:, [0, 2, 1] ]
    return (predicted_stances, predicted_truth, pp_stances, pp_truth, clf_claim)
    


def get_source(url):    
    url = str(url)
    try:
        return tld.get_tld(url)
    except:
        return url
        


def snopes_pp(data):
    """
    pre-process snopes data
    """
    # get unique claims
    claims = pd.Series(data.claimHeadline).unique() 
    #claim_tab = pd.DataFrame({'claimId': claims, 'claimCount': range(1, len(claims)+1)})
    dic_claims = {c: (i+1) for i, c in enumerate(claims)}
    
    articles = pd.Series(data.articleHeadline).unique()
    #article_tab = pd.DataFrame({'articleId': articles, 'articleCount': range(1, len(articles)+1)})
    dic_articles = {a: (i+1) for i, a in enumerate(articles)}
    
    sources = pd.Series(data.source).unique()
    #source_tab = pd.DataFrame({'source': sources, 'sourceCount': range(1, len(sources)+1)})
    dic_sources = {s: (i+1) for i, s in enumerate(sources)}
    
    data = data.assign(claimCount   = data_pp.apply_dic(data, dic_claims, 'claimHeadline'))
    data = data.assign(articleCount = data_pp.apply_dic(data, dic_articles, 'articleHeadline'))
    data = data.assign(sourceCount  = data_pp.apply_dic(data, dic_sources, 'source'))        
    data = data.drop_duplicates()
    
    return data
        

def snopes_read(loc=''):
    fn = 'GoogleResultsWithLinks.csv'
    a = pd.read_csv(loc + fn)
    
    strip = lambda s: str(s).strip()
    
    a.ArticleHeadline = map(strip, a.ArticleHeadline)
    a['source'] = map(get_source, a.URL)
    
    res = {}
    jloc = loc + 'Snopes/'
    for json_file in os.listdir(jloc):
        if json_file.endswith('.json'):
            with open(jloc + json_file) as jf:
                d = json.load(jf)
                res[d['Claim']] = d['Credibility']
    
    get_vera = lambda c: res[c] if c in res else 'unknown'
    a['claimTruth']  = map(get_vera, a.Claim)
    
    a = a.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1)
    a = a[a.apply(lambda x: x['claimTruth'] in ['true', 'false'], axis=1)]
    
    a = a[a.source != 'snopes.com']
    
    # take only the top sources
    source_freq = a.source.value_counts()
    
    a = a[a.apply(lambda x: x['source'] in source_freq[:1000], axis=1)]
    
    a = a.rename(columns={'Claim': 'claimHeadline', 'ArticleHeadline': 'articleHeadline'})
    a.articleHeadline = map(lambda x: x.decode('utf-8', errors='ignore'), a.articleHeadline)
    a.claimHeadline = map(lambda x: x.decode('utf-8', errors='ignore'), a.claimHeadline)
    
    a['claimId'] = 0
    a['articleId'] = 0
    
    return a

def number_article(data, start = 0):
    articles = pd.Series(data.articleHeadline).unique()
    dic_articles = {a: (i+1+start) for i, a in enumerate(articles)}
    data = data.assign(articleCount = data_pp.apply_dic(data, dic_articles, 'articleHeadline'))
    return data

def snopes_split(a):
    
    #claims = a.claimHeadline.unique()
    #claims_train = claims[:2700]
    #claims_val   = claims[2700:3600]
    #claims_test  = claims[3600:]
    
    claims = a.claimCount.unique()
    claims_train = claims[:2700]
    claims_val   = claims[2700:3600]
    claims_test  = claims[3600:]
    
    a_train = a[a.apply(lambda x: x['claimCount'] in claims_train , axis=1)]
    a_val   = a[a.apply(lambda x: x['claimCount'] in claims_val   , axis=1)]
    a_test  = a[a.apply(lambda x: x['claimCount'] in claims_test  , axis=1)]
    
    # renumber the articles
    a_train.articleCount = 1 + np.arange(len(a_train))
    a_val.articleCount   = 1 + np.arange(len(a_train), len(a_train) + len(a_val), 1)
    a_test.articleCount  = 1 + np.arange(len(a_train) + len(a_val), len(a_train) + \
                                 len(a_val) + len(a_test), 1)
    
    a_train.index = range(len(a_train.index))
    a_val.index   = range(len(a_val.index))
    a_test.index  = range(len(a_test.index))
    
    return (a_train, a_val, a_test)




def snopes_baseline(clf_st):
    #(ps, pt, pp_s, pp_t, clf_st, clf_vera) = util.baseline_crowd(\
    #train_val_data_pp, X_train_val, test_data_pp, X_test, cds, cdv, return_proba=True)
    
    a = snopes_read(loc='../data/')

    a = snopes_pp(a)

    (a_train, a_val, a_test) = util.snopes_split(a)
    
    
    f_train = app.get_features(a_train)
    f_val   = app.get_features(a_val)
    f_test   = app.get_features(a_test)
    
    # set train data = train + val
    a_train = pd.concat((a_train, a_val))
    a_train.index = range(len(a_train))
    f_train = scipy.sparse.vstack((f_train, f_val))
    
    st = clf_st.predict(f_train)
    a_train['articleHeadlineStance'] = st
    a_test['articleHeadlineStance'] ='none'
    
    x = util.baseline(a_train, f_train, a_test, f_test, clf_stance=clf_st, \
                      source_len=1000, return_proba=True, pp_truth_permu=False)
    
    
    #util.get_acc(a_test, x[0], x[1], x[3], bin_brier=True)
    get_acc(a_val, x[0], x[1], x[3], bin_brier=True)
    
    
    
def snopes_cmv():
    cmv = models.model_cv(data_all, X, cds, cdv)
    cmv.init_model()
    cmv.em(5)
    
    st2 = cmv.clf_stance.predict(f_train)
    a_train['articleHeadlineStance'] = st2
    a_test['articleHeadlineStance'] = cmv.clf_stance.predict(f_test)
    
    a_all = pd.concat([a_train, a_test], ignore_index=True)
    f_all = scipy.sparse.vstack((f_train, f_test))
        
    cf = models.model_transfer(a_all, f_all, [], [], n_train=len(a_train), vera_range=[0,1])
    
    cf.init_model(cmv.clf_stance)
    
    cf.em(5)
    
    util.get_acc(a_test, cf.res_s[cf.n_train:], \
                   cf.res_v[cf.train_m:], cf.alpha[cf.train_m:], bin_brier=True)
                   

def export(train_data, X_train, source_len=724):
    """
    produce classifiers for web app using baseline.
    return (claim classifier, stance classifier, 
           a dictionary from source name to source id)
    """
    clf_stance = sklearn.linear_model.LogisticRegression(penalty='l1')
    stances = []
    for s in train_data.articleHeadlineStance:
        if s == 'against':
            stances.append(-1 )
        elif s == 'for':
            stances.append(1)
        else:
            stances.append(0)
    clf_stance.fit(X_train, stances)
    
    dic_f = get_features(train_data, \
            train_data.articleHeadlineStance, source_len=source_len)
        
    (train_claims, train_y) = extract_truth_labels(train_data)
    n = len(train_claims)
    m = dic_f.items()[0][1].shape[0]
    
    train_f = np.zeros((n, m))
    for i, c in enumerate(train_claims): train_f[i, :] = dic_f[c]
    
    
    # only take true/false
    final_train_f = []
    final_train_y = []
    for i in range(n):
        if train_y[i] != 'unknown':
            final_train_f.append(train_f[i,:])
            final_train_y.append(1 if train_y[i] == 'true' else 0)
            
    final_train_f = np.asarray(final_train_f)        
    
    # train the model
    clf_claim = sklearn.linear_model.LogisticRegression(fit_intercept=False)
    clf_claim.fit(final_train_f, final_train_y)
    
    #sources = ['' for i in range(source_len)]
    #for row in train_data.itertuples():
    #    i = row.sourceCount - 1 # 1-index to 0-index
    #    sources[i] = row.source
    sources = get_dic_sources(train_data)
    
    return (clf_claim, clf_stance, sources)
                   
