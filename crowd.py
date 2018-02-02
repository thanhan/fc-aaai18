#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 13:21:39 2017

@author: atn
"""

import pandas as pd
import numpy as np
import collections

def read_data():
    a = pd.read_csv('mturk/Batch_2740824_batch_results.csv')

    g = g = a[['Input.articleCount', 'Answer.stance', 'Input.articleHeadlineStance',\
           'Input.claimHeadline', 'Input.articleHeadline']].groupby('Input.articleCount')
    l = g.agg(lambda x: sorted(tuple(x)))
    #l.columns = ['agg ans', 'a']
    #l = l.reset_index()
    
def split_to_batch(data, bs = 5):
    list_col = []
    for j in range(1, bs+1, 1):
        list_col.append('claimHeadline' + str(j))
        list_col.append('articleHeadline' + str(j))
        list_col.append('articleCount' + str(j))
        list_col.append('articleHeadlineStance' + str(j))
    
    res = pd.DataFrame(columns = list_col)
    n = len(data)
    m = 0 # index in res
    for i in range(0, n, 5):
        new_row = []
        for j in range(i, i+bs, 1):
            if j < n:
                new_row.append(data.iloc[j]['claimHeadline'])
                new_row.append(data.iloc[j]['articleHeadline'])
                new_row.append(data.iloc[j]['articleCount'])
                new_row.append(data.iloc[j]['articleHeadlineStance'])
        if len(new_row) == bs*4:
            res.loc[m] = new_row
        m += 1
        #('claimHeadline' + str(j))
        #('articleHeadline' + str(j))
        
    return res


def split_to_batch_vera(data, bs = 3):
    list_col = []
    for j in range(1, bs+1, 1):
        list_col.append('claimH' + str(j))
        list_col.append('stances' + str(j))
        list_col.append('claimCount' + str(j))
    
    res = pd.DataFrame(columns = list_col)
    n = len(data)
    m = 0 # index in res
    for i in range(0, n, bs):
        new_row = []
        for j in range(i, i+bs, 1):
            if j < n:
                new_row.append(data.iloc[j]['claimH'])
                new_row.append(data.iloc[j]['stances'])
                new_row.append(data.iloc[j]['claimCount'])
        if len(new_row) == bs*3:
            res.loc[m] = new_row
        m += 1
        #('claimHeadline' + str(j))
        #('articleHeadline' + str(j))
        
    return res


def get_dic_aid_stance(data):
    dic = {}
    n = len(data)
    for i in range(n):
        aid = data.iloc[i]['articleCount']
        stance = data.iloc[i]['articleHeadlineStance']
        dic[aid] = stance
        
    return dic

p1 = "mturk/Batch_2765829_batch_results.csv"
def read_batch_data(data, bs = 5, fn = None):
    dic_as = get_dic_aid_stance(data)
    if fn == None:
        fn = 'mturk/Batch_2768981_batch_results.csv'
    a = pd.read_csv(fn)
    n = len(a)

    res = []
    for i in range(n):
        for j in range(1, bs+1, 1):
            aid = a.iloc[i]['Input.articleCount' + str(j)]
            ans = a.iloc[i]['Answer.stance' + str(j)]
            wid = a.iloc[i]['WorkerId']
            gold = dic_as[aid]
            res.append ([aid, ans, wid, gold])

    res = pd.DataFrame(res, columns = ['aid', 'ans', 'wid', 'gold'])
    return res
            
def majority_vote(cd):
    dic = {} # dic aid -> list of labels
    
    for i in range(len(cd)):
        aid = int(cd.iloc[i]['aid'])
        wid = cd.iloc[i]['wid']
        l   = cd.iloc[i]['ans']
        
        if aid not in dic: dic[aid] = ([], [])
        
        dic[aid][0].append(str(l).lower())
        dic[aid][1].append(wid)
    dic_mv = {}
    for aid, (ls, wids) in dic.items():
        c = collections.Counter(ls) 
        dic_mv[aid] = c.most_common()[0][0]
    return dic, dic_mv



class CD:
    """
    Crowd Data
    """
    def __init__(self, data, lab2num = None, labtype = 'stance'):
        self.data = data
        self.dic_al, self.dic_mv = majority_vote(data)
        self.l = len(data)
        
        self.dic_wa = {} # wid -> aid
        for i in range(len(data)):
            aid = int(data.iloc[i]['aid'])
            wid = data.iloc[i]['wid']
            if wid not in self.dic_wa: self.dic_wa[wid] = []
            self.dic_wa[wid].append(aid)
            
        # list of articles and workers
        self.a = sorted(self.dic_al.keys())
        self.w = sorted(self.dic_wa.keys())
        
        
        
        self.na = len(self.a)
        self.nw = len(self.w)
 
        # map labels from strings to codes
        if lab2num == None:
            self.lab2num = {'against': 0, 'observing': 1, 'for': 2, 'nan': -1, \
                            'true': 2, 'unknown': 1, 'false': 0}
            
        if labtype == 'stance':
            self.num2lab = {0: 'against', 1: 'observing', 2: 'for', -1: 'nan'}
        else:
            self.num2lab = {0: 'false', 1: 'unknown', 2: 'true', -1: 'nan'}
            
        self.dic_al2 = {} # dic_al with code for label
        for k, (l, w) in self.dic_al.items():
            self.dic_al2[k] = (map(lambda x: self.lab2num[x], l), w)
            
        # get mv labels
        self.mvl = [self.dic_mv[ai] for ai in self.a]
        self.mvc = map(lambda x: self.lab2num[x], self.mvl)
        
        self.to_code()
        
    def to_code(self):
        """
        set the article/worker/label codes
        """
        ac = map(lambda x: self.a.index(x), self.data.aid)
        self.data = self.data.assign(ac = ac)
        
        wc = map(lambda x: self.w.index(x), self.data.wid)
        self.data = self.data.assign(wc = wc)
        
        lc = map(lambda x: self.lab2num[str(x).lower()], self.data.ans)
        self.data = self.data.assign(lc = lc)
        
        # expert worker
        if 'EXPERT' in self.w:
            self.expert_wid = self.w.index('EXPERT')
        else:
            self.expert_wid = -1
        
    def get_wp(self, dic_gold):
        """
        worker performance
        """
        self.dic_wp = {}
        for i in range(self.l):
            aid = int(self.data.iloc[i]['aid'])
            wid = self.data.iloc[i]['wid']
            lc = self.data.iloc[i]['lc']
            g  = dic_gold[aid]
            if wid not in self.dic_wp: self.dic_wp[wid] = [0, 0]
            self.dic_wp[wid][0] += 1
            if self.lab2num[g.lower()] == lc:
                self.dic_wp[wid][1] += 1


def get_acc(gold_dic, res_a, res_l):
    lab2num = {'against': 0, 'observing': 1, 'for': 2, 'nan': -1}
    s = 0
    c = 0
    for a, l in zip(res_a, res_l):
        s += 1
        g = gold_dic[a].lower()
        g = lab2num[g]
        if g == l: c+= 1
        
    print c, s
    return c * 1.0 / s
        

def get_gold_lab(gold_dic, a):
    lab2num = {'against': 0, 'observing': 1, 'for': 2, 'nan': -1}
    res = []
    for x in a:
        res.append(lab2num[gold_dic[x]])
    return res

class DS:
    """
    Dawid-Skene
    """
    def __init__(self, cd, c = 3, smooth = 0.001, list_expert = []):
        """
        list_expert: list of id for workers with perfect accuracy
        """
        self.cd = cd
        self.c = c      # number of categories
        self.n = cd.na  # number of articles/items
        self.m = cd.nw  # number of workers
        self.smooth = smooth
        self.l = len(cd.data)
        self.list_expert = list_expert
        
        self.init()
        
        
    def init(self, d = 0.6):
        """
        d = value for diagonal
        """
        self.cm = np.ones((self.m, self.c, self.c)) * ((1 - d)/(self.c - 1))
        for w in range(self.m):
            if w in self.list_expert:
                self.cm[w] = np.eye(self.c)
            else:
                for k in range(self.c):
                    self.cm[w][k][k] = d
                    
        self.list_ac = list(self.cd.data.ac)
        self.list_wc = list(self.cd.data.wc)
        self.list_lc = list(self.cd.data.lc)
         
    def e_step(self):
        """
        evaluate posterior over the true value of each instance
        self.pos is indexed by self.cd.a
        """
        self.pos = np.ones((self.n,self.c))
        
        for i in range(self.l):
            #ac = self.cd.data.iloc[i]['ac']
            #wc = self.cd.data.iloc[i]['wc']
            #lc = self.cd.data.iloc[i]['lc']
            ac = self.list_ac[i]
            wc = self.list_wc[i]
            lc = self.list_lc[i]
            self.pos[ac] = self.pos[ac] * self.cm[wc, :, lc]
        for i in range(self.n):
            self.pos[i] = self.pos[i] * 1.0 / np.sum(self.pos[i])
            
            
    def m_step(self):
        self.count = self.smooth * np.ones((self.m, self.c, self.c))
        for i in range(self.l):
            #ac = self.cd.data.iloc[i]['ac']
            #wc = self.cd.data.iloc[i]['wc']
            #lc = self.cd.data.iloc[i]['lc']
            ac = self.list_ac[i]
            wc = self.list_wc[i]
            lc = self.list_lc[i]
            for tl in range(self.c):
                self.count[wc][tl][lc] += self.pos[ac][tl]
        
        
        for j in range(self.m):
            if j in self.list_expert:
                    self.cm[j] = np.eye(self.c)
            else:
                for k in range(self.c):
                    self.cm[j][k] = self.count[j][k] * 1.0 \
                    / np.sum(self.count[j][k])
                
            
        
    def em(self, n_its = 5):
        for it in range(n_its):
            self.e_step()
            self.m_step()
        self.mlc = np.argmax(self.pos, 1)
        self.mll = map(lambda x: self.cd.num2lab[x], self.mlc)
        self.dic_al = {aid: self.mll[i] for i, aid in enumerate(self.cd.a)}
        
        
    def get_dic_aid_pos(self):
        """
        dic: aid -> pos
        """
        self.dic_aid_pos = {}
        for aid, p in zip(self.cd.a, self.pos):
            self.dic_aid_pos[aid] = p
        
        
    def get_full_pos(self, aids):
        """
        pos for all articles in the list aids
        """
        
        self.get_dic_aid_pos()
        res = []
        uniform = np.ones((self.c,)) * (1.0/ self.c)
        for aid in aids:
            if aid not in self.dic_aid_pos:
                res.append(uniform)
            else:
                res.append(self.dic_aid_pos[aid])
                
        self.full_pos = np.asarray(res)
        return self.full_pos
    
    def set_pos(self, pos, aids):
        dic_set = {}
        for p, aid in zip(pos, aids):
            dic_set[aid] = p
        for i, aid in enumerate(self.cd.a):
            self.pos[i] = dic_set[aid]