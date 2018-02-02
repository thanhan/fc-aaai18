#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Data pre-processing and preparation

"""

import pickle
import util
import pandas as pd
import sklearn.linear_model
import numpy as np
import scipy

def get_data_pk():
    f = open('edata.pkl')
    x = pickle.load(f)
    f.close()
    return x

def url_to_source(url):
    try:
        return url.split('/')[2]
    except Exception:
        if url.find('antiviral.gawker.com') > 0: return "antiviral.gawker.com"
        if url.find('twitter') > 0: return "twitter"
        print url
        return -1
        
#  appling a dic to a column
def apply_dic(data, dic, col_name):
    f = lambda row: dic[row[col_name]]
    return data.apply(f, axis = 1)

def process_data(data):
    
    # add url/body/a_stances to data
    (dic_url, dic_body, dic_as) = util.get_dic_aid()
    #aid_url = pd.DataFrame( dic_url.items(), columns = ['articleId', 'url'])
    #data = data.merge(aid_url)
    #data.assign(url = data.apply(lambda row: dic_url[row['articleId']], axis = 1) )
    data = data.assign(url = apply_dic(data, dic_url, 'articleId'))
    data = data.assign(source = data.apply(lambda row: url_to_source(row['url']), axis = 1))
    data = data.assign(body = apply_dic(data, dic_body, 'articleId'))
    data = data.assign(astance = apply_dic(data, dic_as, 'articleId'))
    
    
    # add claim truth label to data
    dic_truth = util.get_dic_truth()
    #cid_truth = pd.DataFrame( dic_truth.items(), columns = ['claimId', 'truth'])
    #data = data.merge(cid_truth)
    #data.assign(claimTruth = data.apply(lambda row: dic_truth[row['claimId']], axis = 1))
    data = data.assign(claimTruth = apply_dic(data, dic_truth, 'claimId'))
    
    # add counts to data
    
    # get unique claims
    claims = pd.Series(data.claimId).unique() 
    #claim_tab = pd.DataFrame({'claimId': claims, 'claimCount': range(1, len(claims)+1)})
    dic_claims = {c: (i+1) for i, c in enumerate(claims)}
    
    articles = pd.Series(data.articleId).unique()
    #article_tab = pd.DataFrame({'articleId': articles, 'articleCount': range(1, len(articles)+1)})
    dic_articles = {a: (i+1) for i, a in enumerate(articles)}
    
    sources = pd.Series(data.source).unique()
    #source_tab = pd.DataFrame({'source': sources, 'sourceCount': range(1, len(sources)+1)})
    dic_sources = {s: (i+1) for i, s in enumerate(sources)}
    
    data = data.assign(claimCount   = apply_dic(data, dic_claims, 'claimId'))
    data = data.assign(articleCount = apply_dic(data, dic_articles, 'articleId'))
    data = data.assign(sourceCount  = apply_dic(data, dic_sources, 'source'))
    
    
    return data
    
    
    
    
def make_stan_input(data, X, data_test = None, X_test = None, mul_lr = False):
      """
      data_test, X_test: include (unlabeled) test data
      mul_lr: input for mul_lr model
      """
      
      # data_all includes train and test data
      if X_test != None:
          data_all = pd.concat([data, data_test], ignore_index = True)
          X_all = scipy.sparse.vstack([X, X_test])
      else:
          data_all = data
          X_all = X
      
      n = data_all.articleCount.max()
      m = 1                         # number of workers
      k = data_all.claimCount.max()
      o = data_all.sourceCount.max()
      
      # make a list of triplets (claim, stance, souce)
      # representing the connections between claims, stances and sources
      # not including claim nor stance labels
      nl = len(data_all)
      list_claim  = data_all.claimCount.values.tolist()
      list_stance = data_all.articleCount.values.tolist()
      list_source = data_all.sourceCount.values.tolist()
      
      # stance labels
      stance_dic = {'against': 1, 'observing': 2, 'for': 3}
      stance_l = map(lambda x: stance_dic[x], data.articleHeadlineStance.values.tolist())
      ns = len(stance_l)
      stance_wid = [1] * ns
      stance_iid = data.articleCount.tolist()
                    
      
      # claim labels
      claim_dic = {'false': 1, 'unknown': 2, 'true': 3}
      #claim_l = data.drop_duplicates(subset = 'claimCount').sort_values('claimCount').claimTruth
      #claim_l = map(lambda x: claim_dic[x], claim_l)
      (claims, claim_l) = util.extract_truth_labels(data)
      claim_l = map(lambda x: claim_dic[x], claim_l)
      nc = len(claim_l)
      claim_wid = [1] * nc
      claim_iid = claims
                    
      # source labels
      no = 1
      source_l = [1]
      source_wid = [1]
      source_iid = [1]
      
      
      #clf = sklearn.linear_model.LogisticRegression(multi_class='multinomial',\
      #                                              solver='lbfgs', C = 1)
      
      clf =  sklearn.linear_model.LogisticRegression(penalty = 'l1')
      
      clf.fit(X, data.articleHeadlineStance)
      stance_mean = clf.intercept_ + X_all.toarray().dot( clf.coef_.T)
      stance_mean = stance_mean[:, [0, 2, 1]]
      
      
      
      res = {'n': n,
             'm': m, 
             'k': k, 
             'o': o, 
             'nl': nl,
             'list_claim': list_claim,
             'list_stance': list_stance, 
             'list_source': list_source, 
             'ns': ns, 
             'stance_l': stance_l,
             'stance_wid': stance_wid, 
             'stance_iid': stance_iid,
             'nc': nc,
             'claim_l': claim_l,
             'claim_wid': claim_wid, 
             'claim_iid': claim_iid,
             'no': no,
             'source_l': source_l,
             'source_wid': source_wid, 
             'source_iid': source_iid, 
             #'c': c,
             'dim_s': 518, 
             'fs': X_all.toarray(),
             'ws': clf.coef_.T,
             'stance_intercept': clf.intercept_,
             'ws_var': 1,
             'stance_mean': stance_mean,
             'source_score': np.zeros((o,)),
             'source_score_var': 2,
             'claim_intercept': np.zeros((3,))
             }

      if mul_lr:
          res['n'] = 1489
          res['fs'] = X.toarray()
      #if X_test != None:
      #    stance_mean_test = clf.intercept_ + X_test.toarray().dot( clf.coef_.T)
      #    res['stance_mean_test'] = stance_mean_test
          
      return res
                        
      
      
      
    