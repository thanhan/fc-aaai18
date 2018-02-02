#!/usr/bin/env python2
# -*- coding: utf-8 -*-
#import pystan
import sklearn
import sklearn.linear_model
import numpy as np
import scipy
import pandas as pd
import util
import collections
import crowd

schools_code = """
data {
    int<lower=0> J; // number of schools
    real y[J]; // estimated treatment effects
    real<lower=0> sigma[J]; // s.e. of effect estimates
}
parameters {
    real mu;
    real<lower=0> tau;
    real eta[J];
}
transformed parameters {
    real theta[J];
    for (j in 1:J)
    theta[j] <- mu + tau * eta[j];
}
model {
    eta ~ normal(0, 1);
    y ~ normal(theta, sigma);
}
"""

schools_dat = {'J': 8,
               'y': [28,  8, -3,  7, -1,  1, 18, 12],
               'sigma': [15, 10, 16, 11,  9, 11, 10, 18]}

#fit = pystan.stan(model_code=schools_code, data=schools_dat,
#                  iter=1000, chains=4)


model1 = """
data {
      int n; // number of items
      int m; // number of workers
      int k; // number of observations (wid, label)
      matrix[3, 3] cm[m]; // confusion matrix, provided as data
      //real f[n]; // features
      int l[k]; // crowd label
      int wid[k]; // worker id
      int iid[k]; // instance id
}
parameters {
    simplex[3] z[n]; // true label
}
model {
       real ps[3];
       for (i in 1:k)
           //l[k] ~ categorical(cm[wid[k][z[iid[k]]]])           
           for (j in 1:3)
               ps[j] = log(z[iid[k]][j]) + log(cm[wid[i]][j][l[i]]);
           target += log_sum_exp(ps);
}
"""


model1_data = {'n': 3, 
               'm': 2, 
               'k': 7,
               'cm': [[ [.8, .1, .1], [.1, .8, .1], [.1, .1, .8] ],
                      [ [.8, .1, .1], [.1, .8, .1], [.1, .1, .8] ]  ], 
               'l': [1, 1, 1, 1, 1, 1, 1], 
               'wid': [1, 2, 1, 1, 2, 2, 1],
               'iid': [1, 1, 1, 1, 1, 1, 1],
                      }

fact_model = """


"""


fact_data = {'n': 2, # number of stances
             'm': 1, # number of workers
             'k': 1, # number of claims
             'o': 1, # number of sources
             
             # triplet of (claim, stance, source)
             'nl': 2,
             'list_claim': [1, 1],
             'list_stance': [1, 2],
             'list_source': [1, 1],
             
             #stance labels
             'ns':          4,   # number of observations (wid, label)
             'stance_l':    [3, 3, 3, 3],   # label
             'stance_wid':  [1, 1, 1, 1],   #worker id
             'stance_iid':  [1, 1, 2, 2],   #stance id
             
             # claim labels
             'nc'        : 1     , # number of observations (wid, label)
             'claim_l'   : [3]   , #label
             'claim_wid' : [1]   , # worker id
             'claim_iid' : [1]   , # instance id
             
             # source labels
             'no':          1  , # number of observations (wid, label)
             'source_l':   [3] , #label
             'source_wid': [1] , # worker id
             'source_iid': [1] , # instance id
             'c':          [-1, 1]
             }


class gibbs_sampler:
    def __init__(self, data, p_s, vera, seed = 1, burn = 500, pc = 1, pv = 1, muc = 0):
        """
        data : include list of (claim, stance, source)
        p_s  : categorial parameter for s (size n x 3) each row = (against, observe, for)
        vera : veracity scores of the claims (could be known or unknown)
        """
        # n: number of stances
        # m: number of claims
        # o: number sources
        self.n = data.articleCount.max()
        self.m = data.claimCount.max()
        self.o = data.sourceCount.max()
        
        self.d = self.process_data(data)
        
        
        self.p_s = np.asarray(p_s)
        self.vera = vera
        self.rs = np.random.RandomState(seed)
        
        self.burn = burn
        
        self.init_pc = pc
        self.init_pv = pv
        self.init_muc = muc
        self.init_sampler()
        
    def process_data(self, data):
        d = data[['claimCount', 'articleCount', 'sourceCount']]
        d = d - 1 # 1-index to 0-index
        d = np.array(d)
        return d
        
        
    def init_sampler(self):
        self.s = np.zeros((self.n,))
        x = self.rs.rand(self.n)
        p_s = self.p_s
        self.s[:] = 1
        self.s[x < p_s[:, 0] + p_s[:, 1]] = 0
        self.s[x < p_s[:, 0]] = -1
        
        self.v = np.zeros((self.m,))
        for i in range(self.m):
            self.v[i] = self.vera[i] if self.vera[i] else 0
            
        self.c = np.zeros((self.o,))
        
        self.pc = self.init_pc
        self.pv = self.init_pv
        self.muc = self.init_muc
        
    def eval_mu_v(self):
        """
        evaluate the mean of V (veracity)
        """
        self.muv = np.zeros((self.m,))
        for claim, stance, source in self.d:
            self.muv[claim] += self.s[stance] * self.c[source]
        
    def update_ps(self):
        """
        update the categorical parameters for s
        """
        p_s = self.p_s.copy()
        map_j = [-1, 0, 1]
        self.eval_mu_v()
        mu = self.muv
        sigma_v = pow(1/self.pv, 0.5)
        
        for claim, stance, source in self.d:
            i = stance
            for j in range(3):
                new_mu = mu[claim] - self.s[i] * self.c[source] + map_j[j] * self.c[source]
                p_s[i][j] = p_s[i][j] * scipy.stats.norm.pdf(self.v[claim], new_mu,\
                   sigma_v)
            # normalize for p_s to sum to 1
            if np.sum(p_s[i]) <=0: 
                p_s[i] = self.p_s[i].copy()
                #print claim, stance, source
            p_s[i] = p_s[i] * 1.0 / np.sum(p_s[i])
            
        return p_s
        
    def eval_pos_c(self):
        """
        evaluate the posterior for c
        """
        # posterior mean and precision
        self.pc_m = np.zeros((self.o,))
        self.pc_p = np.zeros((self.o,))
        
        self.eval_mu_v()
        sum_sx = np.zeros((self.o,))
        sum_ss = np.zeros((self.o,))
        
        for claim, stance, source in self.d:
            sum_ss[source] += self.s[stance] * self.s[stance]
            sum_sx[source] += self.s[stance] * (self.v[claim] - self.muv[claim] + \
                  self.s[stance] * self.c[source])
        
        for source in range(self.o):
            self.pc_p[source] = self.pc + self.pv * sum_ss[source]
            self.pc_m[source] = (self.pc * self.muc + self.pv * \
                     sum_sx[source]) / self.pc_p[source]
        
    def sample_c(self):
        # posterior precision
        self.make_f()
        self.pc_p = self.pc * np.eye(self.o) + self.pv * self.f.T.dot(self.f)
        # posterior mean
        inv_p = np.linalg.inv(self.pc_p)
        mu0 = np.ones(self.o) * self.muc # prior mean
        self.pc_m = inv_p.dot(self.pc * np.eye(self.o).dot(mu0)  + \
                              self.pv * self.f.T.dot(self.v))
        
        self.c = self.rs.multivariate_normal(self.pc_m, inv_p)
        #self.c = self.pc_m
        
        
    def sample(self, nit = 2000):
        self.save_s = np.zeros((nit, self.n))
        self.save_v = np.zeros((nit, self.m))
        self.save_c = np.zeros((nit, self.o))
        self.save_pc = np.zeros((nit,))
        self.save_pv = np.zeros((nit,))
        self.save_muc = np.zeros((nit,))
        self.save_muv = np.zeros((nit, self.m))
        
        # sample S
        
        
        for it in range(nit):
            
            p_s = self.update_ps()
            x = self.rs.rand(self.n)
            self.s[:] = 1
            self.s[x < p_s[:, 0] + p_s[:, 1]] = 0
            self.s[x < p_s[:, 0]] = -1
            
            # sample V
            self.eval_mu_v()
            sd_v = pow( 1.0 / self.pv, 0.5)
            for claim in range(self.m):
                if self.vera[claim] != None:
                    self.v[claim] = self.vera[claim]
                else:
                    self.v[claim] = self.rs.normal(self.muv[claim], sd_v)
            
            #sample C
            self.sample_c()
                
            # update pv, pc, muc
            
            self.eval_mu_v()
            ss_v = 0; 
            for claim in range(self.m): ss_v += pow(self.v[claim] - self.muv[claim], 2)
            self.pv = self.rs.gamma(self.m/2.0 + 1, 1.0 / (ss_v/2.0) )
            
            ss_c = np.sum(pow(self.c - self.muc, 2))
            self.pc = self.rs.gamma(self.o/2.0 + 1, 1.0 / (ss_c/2.0) )
            
            self.muc = self.rs.normal(sum(self.c) * 1.0 / self.o, \
                                      pow(1.0 / (self.pc*self.o), 0.5)  )
            
            # save samples
            self.save_s[it, :] = self.s.copy()
            self.save_v[it, :] = self.v.copy()
            self.save_c[it, :] = self.c.copy()
            self.save_pv[it] = self.pv
            self.save_pc[it] = self.pc
            self.save_muc[it] = self.muc
            self.save_muv[it, :] = self.muv
            
            
    def map_stance(self):
        a = self.save_s[self.burn:, :]
        res = []
        dic_s = {-1: 'against', 0: 'observing', 1: 'for'}
        for i in range(self.n):
            ct = collections.Counter(a[:, i])
            res.append(dic_s[int(ct.most_common()[0][0])])
        return res
        
    def map_veracity(self, t = 0.5):
        a = self.save_v[self.burn:, :]
        b = np.mean(a, 0)
        
        res = []
        for i in range(self.m):
            if b[i] < -t: 
                r = 'false'
            elif b[i] < t:
                r = 'unknown'
            else:
                r = 'true'
            res.append(r)
            
        return res
    
    def check_dup(self):
        self.check = np.zeros((self.m, self.o))
        for claim, stance, source in self.d:
            if self.check[claim, source] != 0:
                print claim, stance, source
            self.check[claim, source] = stance
    
    
    
    def make_f(self):
        self.f = np.zeros((self.m, self.o))
        for claim, stance, source in self.d:
            self.f[claim, source] += self.s[stance]
        
        
def baseline_gibbs(train_data, X_train, test_data, X_test, t = 0.5):
    clf_stance = sklearn.linear_model.LogisticRegression()
    clf_stance.fit(X_train, map_stance_label(train_data.articleHeadlineStance))
    p = clf_stance.predict(X_test)
    
    (data_all, p_s, vera) = run(train_data, X_train, test_data, X_test)
    gs = gibbs_sampler(data_all, p_s, vera)
    
    gs.s[-len(p):] = p
    # make features:
    f = np.zeros((gs.m, gs.o))
    for claim, stance, source in gs.d:
        f[claim, source] += gs.s[stance]
    
    #import mord
    #reg = mord.LAD()
    reg = sklearn.linear_model.LinearRegression(fit_intercept = False)
    #reg = sklearn.linear_model.LogisticRegression(fit_intercept = False)
    reg.fit(f[:180], vera[:180])
    b = reg.predict(f[180:])
    res = []
    for i in range(60):
        if b[i] < -t: 
            r = 'false'
        elif b[i] < t:
            r = 'unknown'
        else:
            r = 'true'
        res.append(r)

    return res
    
dics = {'against': -1, 'observing': 0, 'for': 1}
dicv = {'false': -1, 'unknown': 0, 'true': 1}
    
def map_stance_label(l):
    return [dics[x] for x in l]
        
def run(train_data, X_train, test_data, X_test, em_it = 3, return_data = True):
    clf_stance = sklearn.linear_model.LogisticRegression()
    clf_stance.fit(X_train, map_stance_label(train_data.articleHeadlineStance))
    
    data_all = pd.concat([train_data, test_data], ignore_index = True)
    #X_all = scipy.sparse.vstack([X_train, X_test])
    
    # prepare data for gibbs sampler
    pt = clf_stance.predict_proba(X_test)
    p_s = np.zeros((X_train.shape[0], 3))
    for i, s in enumerate(train_data.articleHeadlineStance):
        j = dics[s] + 1
        p_s[i][j] = 1.0
    p_s = np.vstack((p_s, pt))
    
    vera = util.extract_truth_labels(train_data)[1]
    test_claims = sorted(test_data.claimCount.unique().tolist())
    vera = [dicv[x] for x in vera]
    vera.extend([None] * len(test_claims))
    
    if return_data:
        return (data_all, p_s, vera)
    
    for it in range(em_it):
        # E-step: sampling posterior
        gs = gibbs_sampler(data_all, p_s, vera)
        # M-step: fit clf
        pass
    
class model:
    def __init__(self, train_data, X_train, test_data, X_test, seed = 1, \
                 sample_its = 2000, burn = 500):
        self.data_all = pd.concat([train_data, test_data], ignore_index = True)
        self.n = self.data_all.articleCount.max()
        self.m = self.data_all.claimCount.max()
        self.o = self.data_all.sourceCount.max()
        
        self.n_train = len(train_data)
        self.n_test  = len(test_data)
        
        self.train_data = train_data
        self.test_data  = test_data
        self.X_train = X_train
        self.X_test  = X_test
        self.X = scipy.sparse.vstack((X_train, X_test))
        
        self.d = self.process_data(self.data_all)
        
        self.rs = np.random.RandomState(seed = seed)
        self.burn = burn
        
        self.dics = {'against': 0, 'observing': 1, 'for': 2}
        self.dicv = {'false': 0, 'unknown': 1, 'true': 2}
        
        #self.tw = tw # weight for test data
        self.sample_its = sample_its
        self.burn = burn
        
        
        
    def process_data(self, data):
        d = data[['claimCount', 'articleCount', 'sourceCount']]
        d = d - 1 # 1-index to 0-index
        d = np.array(d)
        return d
    
    
    
    def init_model(self):
        # fit stance and vera clf only on train data
        self.clf_stance = sklearn.linear_model.LogisticRegression(penalty = 'l1')
        #self.clf_stance = sklearn.linear_model.SGDClassifier(loss = 'log', n_iter = 5000)
        train_stances = [self.dics[x] for x in self.train_data.articleHeadlineStance]
        self.clf_stance.fit(self.X_train, train_stances)
        
        
        # p_s = prob of stance using "labels"
        self.ps = np.zeros((self.n_train + self.n_test, 3))
        self.s = np.zeros((self.n_train + self.n_test,))
        test_stances = self.clf_stance.predict(self.X_test)
        for i, s in enumerate(self.train_data.articleHeadlineStance):
            j = self.dics[s]
            self.ps[i][j] = 1.0
            self.s[i] = j
        for i in range(self.n_train, self.n_train + self.n_test, 1):
            self.ps[i, :] = [1.0/3] * 3
            self.s[i] = test_stances[i - self.n_train]
        
        #self.s = self.cat_sample(self.ps)
        self.make_features()
        
        self.vera = util.extract_truth_labels(self.train_data)[1]
        self.vera = [self.dicv[x] for x in self.vera]
        self.train_m = len(self.vera)
        self.clf_vera = sklearn.linear_model.LogisticRegression()
        self.clf_vera.fit(self.f[:self.train_m], self.vera)
        test_claims = sorted(self.test_data.claimCount.unique().tolist())
        self.test_m = len(test_claims)
        
        self.pv = np.zeros((self.train_m + self.test_m, 3))
        for i, j in enumerate(self.vera):
            self.pv[i][j] = 1.0
        for i in range(self.train_m, self.train_m + self.test_m):
            self.pv[i,:] = [1.0/3] * 3
            
        self.v = self.cat_sample(self.pv)
        
        
    def cat_sample(self, p):
        n = len(p)
        res = np.zeros((n,))
        x = self.rs.rand(n)
        
        res[:] = 2
        res[x < p[:, 0] + p[:, 1]] = 1
        res[x < p[:, 0]] = 0
        
        return res
        
    
    def make_features(self):
        self.f = np.zeros((self.m, self.o))
        #w = [-1, 0, 1]
        for claim, stance, source in self.d:
            self.f[claim, source] += (self.s[stance] - 1) #  0,1,2 -> -1, 0 1
            #for j in range(3):
                #self.f[claim, source] += w[j] * self.ps[stance, j]
        #self.f = scipy.sparse.csr_matrix(self.f)
            
    
    def normalize(self, p):
        m = p.shape[1]
        s = np.sum(p, 1)
        s = s.reshape((s.shape[0], 1))
        s = np.repeat(s, m, 1)
        return p / s
    
    def sample_s(self, ps):
        """
        ps = (updated) prior for s
        """
        intercept = self.clf_vera.intercept_
        w = self.clf_vera.coef_.T
        
        res = np.ones((self.n, 3))
        for claim, stance, source in self.d:
            self.f[claim, source] -= (self.s[stance] - 1)
            x = intercept + self.f[claim].dot(w) # if sparse matrix, need to take first row
            #x -= (self.s[stance] - 1) * w[source] #
            
            # res = prob of V given each value of S
            res[stance, 0] = util.softmax(x - w[source])[int(self.v[claim])]
            res[stance, 1] = util.softmax(x            )[int(self.v[claim])]
            res[stance, 2] = util.softmax(x + w[source])[int(self.v[claim])]
            
            prob = res[stance] * ps[stance]
            prob = prob / (np.sum(prob))
            
            # sample new S
            self.s[stance]= self.rs.choice(range(3), p = prob)
            
            # replace S
            self.f[claim, source] += (self.s[stance] - 1)
            
            
        return res
    
    def sample_vs(self):
        """
        Gibbs sampling for veracity and stance
        """
        n_its = self.sample_its
        # classifiers for each iteration of Gibbs
        #self.sgd_stance = sklearn.linear_model.SGDClassifier(loss = 'log')
        self.it_stance = sklearn.linear_model.LogisticRegression(penalty = 'l1')
        #self.sgd_vera = sklearn.linear_model.SGDClassifier(loss = 'log')
        self.it_vera = sklearn.linear_model.LogisticRegression()
        
        # vars to save Gibbs samples
        self.save_s = np.zeros((n_its, self.n))
        self.save_v = np.zeros((n_its, self.m))
        
        self.save_clf_stance_it = np.zeros((n_its, 3))
        self.save_clf_stance_co = np.zeros((n_its, 3, 518)) # 518 = # text features
        
        self.save_clf_vera_it = np.zeros((n_its, 3))
        self.save_clf_vera_co = np.zeros((n_its, 3, self.o))
        
        
        for it in range(n_its):
            # sample V
            if it % 100 == 0: print it,
            # calculate p(v| ...)
            self.make_features()
            pf = self.clf_vera.predict_proba(self.f)
            pos_v = pf * self.pv
            pos_v = self.normalize(pos_v)
            self.v = self.cat_sample(pos_v)
            
            # sample S
            #pos_s = self.ps.copy()
            pf = self.clf_stance.predict_proba(self.X)
            #ps_v  = self.update_ps()
            #pos_s = pos_s * pf * ps_v
            #pos_s = self.normalize(pos_s)
            #self.s = self.cat_sample(pos_s)
            
            ps = self.ps * pf
            self.sample_s(ps)
            
            self.save_s[it, :] = self.s
            self.save_v[it, :] = self.v
            # train clf for vera and stance and save
            if it > self.burn:
                self.make_features()
                #v_w = [1] * self.train_m + [self.tw] * self.test_m
                #s_w = [1] * self.n_train + [self.tw] * self.n_test
                for sgd_it in range(1):
                    #self.sgd_vera.partial_fit(self.f, self.v, [0,1,2], sample_weight = v_w)
                    #self.sgd_stance.partial_fit(self.X, self.s, [0,1,2], sample_weight = s_w)
                    self.it_vera.fit(self.f, self.v)
                    self.it_stance.fit(self.X, self.s)
                    
                    self.save_clf_stance_it[it, :] = self.it_stance.intercept_.copy()
                    self.save_clf_stance_co[it, :, :] = self.it_stance.coef_.copy()
                    
                    self.save_clf_vera_it[it, :] = self.it_vera.intercept_.copy()
                    self.save_clf_vera_co[it, :, :] = self.it_vera.coef_.copy()
        
        
    def m_step(self):
        self.clf_stance.intercept_ = np.mean(self.save_clf_stance_it[self.burn:, ], 0)
        self.clf_stance.coef_ = np.mean(self.save_clf_stance_co[self.burn:, :, :], 0)
        
        self.clf_vera.intercept_ = np.mean(self.save_clf_vera_it[self.burn:, :], 0)
        self.clf_vera.coef_ = np.mean(self.save_clf_vera_co[self.burn:], 0)
        
        
    def e_step(self):
        self.sample_vs()
        
        
    def em(self, em_its = 5):
        """
        M-step: solve for stance/veracity classification
        under the expected labels
        """
        for it in range(em_its):
            self.e_step()
            #self.clf_stance = self.sgd_stance
            #self.clf_vera   = self.sgd_vera
            self.m_step()
            
        # save prediction for stance/vera
        self.res_s = self.map_stance()
        self.res_v = self.map_veracity()
            
    def num2_stance(self, l):
        dics = {0: "against", 1: "observing", 2: "for"}
        return [dics[x] for x in l]
    
    def num2_vera(self, l):
        dicv = {0: "false", 1: "unknown", 2: "true"}
        return [dicv[x] for x in l]
            
    def map_stance(self):
        """
        return most common stance for each stance var
        over the samples.
        """
        a = self.save_s[self.burn:, :]
        res = []
        dic_s = {0: 'against', 1: 'observing', 2: 'for'}
        for i in range(self.n):
            ct = collections.Counter(a[:, i])
            res.append(dic_s[int(ct.most_common()[0][0])])
        return res
        
    def map_veracity(self, t = 0.5):
        a = self.save_v[self.burn:, :]
        dic_v = {0: 'false', 1: 'unknown', 2: 'true'}
        
        res = []
        for i in range(self.m):
            ct = collections.Counter(a[:, i])
            res.append(dic_v[int(ct.most_common()[0][0])])
            
        return res
    
    
def run_em(train_data_pp, X_train , val_data_pp, X_val, seed = 1):
    m = model(train_data_pp, X_train, val_data_pp, X_val, seed = 1, \
                     sample_its = 2000, burn = 1000)
    m.init_model()
    
    for i in range(5):
        print i
        m.em(1)
        res_s = m.map_stance()
        res_v = m.map_veracity()
        print util.get_acc(val_data_pp, res_s[1489:], res_v[180:])
        
        res_v1 = m.num2_vera(m.clf_vera.predict(m.f))
        res_s1 = m.num2_stance(m.clf_stance.predict(m.X))
        print util.get_acc(val_data_pp, res_s1[1489:], res_v1[180:])


def get_expert_df(data, expert_range):
    new_cdata = []
    for i in range(len(data)):
        aid = data.iloc[i]['articleCount']
        if aid in expert_range: 
            lab = data.iloc[i]['articleHeadlineStance']
            new_cdata.append([aid, lab, 'EXPERT', lab])
        
    new_cdata = pd.DataFrame(new_cdata, columns = ['aid', 'ans', 'wid', 'gold'])
    return new_cdata

def prepare_cm_data(train_data, X_train, test_data, X_test, cdata, \
                    expert_range = [], train_range = 1489,\
                    test_range = 2071):
    """
    cdata: raw crowd data for stances
    expert: range of article id to have expert label
    output cds and cdv
    """
    data_all = pd.concat([train_data, test_data], ignore_index = True)
    X = scipy.sparse.vstack((X_train, X_test))
 
    # append expert data to cdata:      
    
    
    # take train portion in crowd data
    cdata_train = cdata[cdata.aid <= train_range]
    cdata_test  = cdata[(cdata.aid > train_range) & (cdata.aid <= test_range)]
    
    expert_train = get_expert_df(train_data, expert_range)
    train_cdata = pd.concat([cdata_train, expert_train], ignore_index = True)
    
    expert_test = get_expert_df(test_data, expert_range)
    test_cdata = pd.concat([cdata_test, expert_test], ignore_index = True)
    
    # build veracity data
    vera = util.extract_truth_labels(train_data)[1]
    n = len(vera)
    
    vera_data = pd.DataFrame({'aid': range(1, n+1, 1),\
                        'ans': vera, \
                        'wid': 'EXPERT'})
    
    cds = crowd.CD(train_cdata)
    cdv = crowd.CD(vera_data, labtype='vera')
    
    cds_test = crowd.CD(test_cdata)
    return (data_all, X, cds, cdv, cds_test)
    

class crowd_model(model):
    def __init__(self, data_all, X, cds, cdv, seed = 1, sample_its = 2000, \
                 burn = 500, n_train = 2071, vera_range=[0,1,2]):
        """
        cds: crowd data for stances
        cdv: crowd data for veracity
        
        data_all, X: data and features for both train and test.
        n_train: number of train instances (placed before test instances)
        
        inference by Gibbs sampling
        """
        self.data_all = data_all
        self.X = X
        self.cds = cds
        self.cdv = cdv
        
        self.n = self.data_all.articleCount.max()
        self.m = self.data_all.claimCount.max()
        self.o = self.data_all.sourceCount.max()
        
        
        self.d = self.process_data(self.data_all) # in 0-index
        
        self.rs = np.random.RandomState(seed = seed)
        
        self.dics = {'against': 0, 'observing': 1, 'for': 2}
        self.dicv = {'false': 0, 'unknown': 1, 'true': 2}
        
        #self.tw = tw # weight for test data
        self.sample_its = sample_its
        self.burn = burn
        self.n_train = n_train
        
        self.vera_range = vera_range
        
    def init_model(self):
        """
        init the stance and claim vera classifier
        """
        self.dss = crowd.DS(self.cds, list_expert=[self.cds.expert_wid])
        self.dsv = crowd.DS(self.cdv, list_expert=[self.cdv.expert_wid])
        
        self.dss.em(10)
        #print "dss 5"
        self.dsv.em(1)
        
        # train the stance clf
        self.clf_stance = sklearn.linear_model.LogisticRegression(penalty = 'l1')
        n = self.n_train # number of stances existing crowd labels
        # assume that articles with lables are at the top
        X3 = scipy.sparse.vstack((self.X[:n], self.X[:n], self.X[:n]))
        #X3 = scipy.sparse.vstack((self.X, self.X, self.X))
        #y = np.asarray([0]*self.n + [1] * self.n + [2] * self.n)
        y = np.asarray([0] * n + [1] * n + [2] * n)
        
        self.aids = list(self.data_all.articleCount)
        self.dss.get_full_pos(self.aids[:n])
        weights = self.dss.full_pos.flatten(order = 'F')
        #print X3.shape, y.shape, weights.shape
        self.clf_stance.fit(X3, y, sample_weight = weights)
        
        #self.clf_stance.fit(self.X[:n], self.dss.mlc)
        
        #print "trained"
        
        # self.ps = factor for s from crowd labels
        # self.s  = initial values for s
        self.dss.get_full_pos(self.aids)
        self.n_test  = self.n - self.n_train
        clf_proba = self.clf_stance.predict_proba(self.X)
        self.ps = self.dss.full_pos 
        self.s = np.argmax(self.ps * clf_proba, 1)
        
        # self.pv = factor for v from crowd labels
        # self.v  = initial value for v
        # assume all labels are from expert
        self.make_features()
        
        self.train_m = len(self.dsv.pos)
        self.clf_vera = sklearn.linear_model.LogisticRegression()
        self.vera = np.argmax(self.dsv.pos, 1)
        self.clf_vera.fit(self.f[:self.train_m], self.vera )
        
        self.pv = np.zeros((self.m, 3))
        for i, j in enumerate(self.vera):
            self.pv[i][j] = 1.0
        for i in range(self.train_m, self.m):
            self.pv[i,:] = [1.0/3] * 3
            
        self.v = self.cat_sample(self.pv)
        
        
    def get_dist(self, save):
        """
        empirical distribution from samples
        """
        a = save[self.burn:, :]
        s0 = np.sum(a == 0, 0)
        s1 = np.sum(a == 1, 0)
        s2 = np.sum(a == 2, 0)
        sa = np.sum([s0, s1, s2], 0) + 1e-9
        
        s0 = s0 / sa
        s1 = s1 / sa
        s2 = s2 / sa
        
        return np.vstack((s0, s1, s2)).T
        
        
    def m_step(self):
        model.m_step(self)
        
        # estimate confution matrices
        #a = self.save_s[self.burn:, :self.n_train]
        dist_s = self.get_dist(self.save_s)
        
        # set posterior to be empirical distribution over Gibbs samples
        self.dss.set_pos(dist_s, self.aids)
        # run dss M-step and E-step
        self.dss.m_step()
        self.dss.e_step()
        # update self.ps
        #uniform = 1.0/3 * np.ones((self.n - self.n_train, 3))
        #self.ps = np.vstack((self.dss.pos, uniform))
        self.ps = self.dss.get_full_pos(self.aids)
        
        dist_v = self.get_dist(self.save_v)
        self.pos_v = dist_v
        
    def get_res(self, n_train = None, n_train_vera = None):
        """
        return ps, pt, pp_s, pp_t for test instances
        """
        
        ps = self.map_stance()[n_train:]
        pt = self.map_veracity()[n_train_vera:]
        
        pp_s = self.get_dist(self.save_s)[n_train:, :]
        pp_t = self.get_dist(self.save_v)[n_train_vera:, :]
        
        return (ps, pt, pp_s, pp_t)
        
    
class model_cv(crowd_model):
    """
    crowd model with inference by variational method
    """
    
    #def __init__(self, data_all, X, cds, cdv, seed = 1, sample_its = 2000, \
    #             burn = 500, n_train = 2071):
    #    crowd_model.__init__(self, data_all, X, cds, cdv, seed = 1, sample_its = 2000, \
    #             burn = 500, n_train = 2071)
    #    self.vera_range = [0,1,2]
    
    def make_expected_features(self):
        self.ef = np.zeros((self.m, self.o))
        for claim, stance, source in self.d:
            self.ef[claim, source] += (self.es[stance]) # es is from -1 to 1
    
    def eval_es(self):
        """
        evaluate E(S) under q
        """
        c = np.asarray([[-1], [0], [1]])
        self.es = self.beta.dot(c)
        
    
    def eval_elog_pv(self):
        """
        evaluate E log p(V_i | S, C) under q(S) for each V_i
        """
        self.eval_es()
        self.make_expected_features()
        
        self.epredictv = self.clf_vera.predict_proba(self.ef)
        self.elpv = np.log(self.epredictv)
    
    def e_step(self, n_it = 5):
        """
        variational approximation to the posterior p(V, S | Data, params)
        is prod_i q(V_i) prod_ij q(S_ij)
        where:
            q(V_i)   = Cat(alpha)
            q(S_ij)  = Cat(beta)
        """
        #self.alpha = 1.0/3 * np.ones((self.m, 3))
        self.alpha = self.pv.copy()
        #self.beta  = 1.0/3 * np.ones((self.n, 3))
        self.beta  = self.clf_stance.predict_proba(self.X) * self.ps
        
        
        for it in range(n_it):
            # update q(V)
            self.eval_elog_pv()
            self.alpha = self.epredictv.copy() * self.pv
            for i in range(self.m): 
                self.alpha[i] = self.alpha[i] / np.sum(self.alpha[i])
            
            #update q(S)
            pfs = self.clf_stance.predict_proba(self.X) * self.ps
            # ep = expected features * weight + intercept
            self.ep = self.clf_vera.intercept_ + self.ef.dot(self.clf_vera.coef_.T)
            for claim, stance, source in self.d:
                # x = ep without this stance
                x = self.ep[claim] - self.ef[claim, source] * \
                    self.clf_vera.coef_[:, source]
                for s_val in [0, 1, 2]:
                    # xs: assume stance is s_val
                    xs = x + self.clf_vera.coef_[:, source] * (s_val - 1)
                    # apply softmax
                    sxs = np.exp(xs) / np.sum(np.exp(xs))
                    if len(sxs) == 1: #binary case
                        es = np.log(sxs).dot(self.alpha[claim, 1])
                    else:
                        es = np.log(sxs).dot(self.alpha[claim])
                    self.beta[stance][s_val] = np.exp( es + np.log(pfs[stance,s_val]) )
                
                #normalize
                self.beta[stance] = self.beta[stance] * 1.0 / np.sum(self.beta[stance])
                    
                    
                    
    def m_step(self):
        """
        update the classifier and the crowd confusion matrix
        """
        X3 = scipy.sparse.vstack((self.X, self.X, self.X))
        n = self.n
        y = np.asarray([0]*n + [1] * n + [2] * n)
        weights = self.beta.flatten(order = 'F')
        self.clf_stance.fit(X3, y, sample_weight = weights)
        
        self.eval_es()
        self.make_expected_features()
        f3 = np.vstack((self.ef, self.ef, self.ef))
        m = self.m
        y = np.asarray([0]*m + [1] * m + [2] * m)
        weights = self.alpha.flatten(order = 'F')
        self.clf_vera.fit(f3, y, sample_weight = weights)
        
        self.dss.set_pos(self.beta, self.aids)
        self.dss.m_step()
        self.dss.e_step()
        #uniform = 1.0/3 * np.ones((self.n - self.n_train, 3))
        #self.ps = np.vstack((self.dss.pos, uniform))
        self.ps = self.dss.get_full_pos(self.aids)
        
        
    def map_stance(self):
        res = []
        dic_s = {0: 'against', 1: 'observing', 2: 'for'}
        for i in range(self.n):
            j = np.argmax(self.beta[i])
            res.append(dic_s[j])
        return res
        
    def map_veracity(self, t = 0.5):
        dic_v = {0: 'false', 1: 'unknown', 2: 'true'}        
        res = []
        for i in range(self.m):
            j = np.argmax(self.alpha[i])
            res.append(dic_v[j])
            
        return res
    
    def get_res(self, n_train = None, n_train_vera = None):
        """
        return ps, pt, pp_s, pp_t for test instances
        """
        
        ps = self.map_stance()[n_train:]
        pt = self.map_veracity()[n_train_vera:]
        
        pp_s = self.beta[n_train:, :]
        pp_t = self.alpha[n_train_vera:, :]
        
        return (ps, pt, pp_s, pp_t)
    
    
    def get_prob(self, cid = 270):
        """
        plot of of articles for a claim
        """
        
        probs = []
        reps = []
        ss = []
        
        max_rep = np.max( np.sum(np.abs(self.clf_vera.coef_), 0) )
        print max_rep
        
        for i in range(self.n):
            if self.d[i][0] == cid:
                source_id = self.d[i][2]
                source = self.data_all.iloc[i]['source']
                probs.append(self.beta[i])
                r = self.clf_vera.coef_[:, source_id]
                rep = np.sum(np.abs(r))
                reps.append(rep / max_rep)
                ss.append(source)
        
        
        return (probs, reps, ss, self.alpha[cid])
                
                
    
class model_transfer(model_cv):
    """
    for transfering (e.g. from Emergent to Snopes data)
    """
    def init_model(self, clf_stance):
        self.clf_stance = clf_stance
        self.s = self.data_all.articleHeadlineStance.tolist()
        
        
        self.make_features()
        
        self.clf_vera = sklearn.linear_model.LogisticRegression()
        
        (claims, vera) = util.extract_truth_labels(self.data_all.iloc[:self.n_train])
        self.train_m = len(claims)
        
        self.vera = [0 if v == 'false' else 1 for v in vera]
        self.clf_vera.fit(self.f[:self.train_m], self.vera )
        
        self.ps = np.ones((self.n, 3))
        self.pv = np.ones((self.m, 2))
        
        
    def m_step(self):
        """
        update the classifier and the crowd confusion matrix
        """
        X3 = scipy.sparse.vstack((self.X, self.X, self.X))
        n = self.n
        y = np.asarray([0]*n + [1] * n + [2] * n)
        weights = self.beta.flatten(order = 'F')
        self.clf_stance.fit(X3, y, sample_weight = weights)
        
        self.eval_es()
        self.make_expected_features()
        f3 = np.vstack((self.ef, self.ef))
        m = self.m
        y = np.asarray([0]*m + [1] * m)
        weights = self.alpha.flatten(order = 'F')
        self.clf_vera.fit(f3, y, sample_weight = weights)        
    
    
def get_claim_slug():
    a = pd.read_csv('emergent/url-versions-2015-06-14.csv')
    res = {}
    for i in range(len(a)):
        cid = a.iloc[i]['claimId']
        csl = a.iloc[i]['claimSlug']
        res[cid] = csl
    return res
    
def get_source_prob(cmv, source_id = 269, dic_slug = None):
    cs = []
    probs = []
    vera = []
    
    for claim, stance, source in cmv.d:
        if source == source_id:
            cid = cmv.data_all.iloc[stance]['claimId']
            cs.append(dic_slug[cid])
            probs.append(cmv.beta[stance])
            vera.append(cmv.alpha[claim])
            
    return (cs, probs, vera)


def take_subset(cs, probs, vera, ss):
    cs = np.asarray(cs)[ss]
    probs = np.asarray(probs)[ss, :]
    vera = np.asarray(vera)[ss, :]
    
    return (cs, probs, vera)
            

def plot_source(cs, probs, vera, save_name = 'abc.pdf'):
    import matplotlib
    #matplotlib.rcParams['pdf.fonttype'] = 42
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    import seaborn as sns
    
    colors = sns.color_palette('colorblind')
    
    m = len(probs)
    
    plt.figure(figsize=(8,1.5))
    
    gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1])
    plt.subplot(gs[0]) 
    
    plt.xlabel('Stance Probability')
    plt.yticks( np.arange(m) + 0.0, cs)

    probs = np.asarray(probs)    
    p1 = plt.barh(range(m), probs[:, 0], color = colors[0], hatch = '//')
    p2 = plt.barh(range(m), probs[:, 1], left = probs[:, 0], color = colors[1])
    p3 = plt.barh(range(m), probs[:, 2], left = probs[:, 0] \
                  + probs[:, 1], color = colors[2], hatch = '\\\\')
    
    plt.legend((p1[0], p2[0], p3[0]), ('Against/False', 'Observing/Unknown', 'For/True'), 
               bbox_to_anchor=(0.2, 1.001, 0.5, 0.01), ncol = 3)
    
    
    plt.subplot(gs[1])
    plt.yticks( np.arange(m) + 0.0, ['']*m)
    plt.xlabel('Claim Veracity Prob.')
    
    vera = np.asarray(vera)
    p1 = plt.barh(range(m), vera[:, 0], color = colors[0], hatch = '//')
    p2 = plt.barh(range(m), vera[:, 1], left = vera[:, 0], color = colors[1])
    p3 = plt.barh(range(m), vera[:, 2], left = vera[:, 0] \
                  + vera[:, 1], color = colors[2], hatch = '\\\\')
    
    
    
    plt.savefig(save_name, bbox_inches = 'tight', pad_inches = 0.001, dpi = 300)
            
        
                
def plot_probs(probs, reps, ss, vera, save_name = 'abc.pdf'):
    import matplotlib
    #matplotlib.rcParams['pdf.fonttype'] = 42
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    import seaborn as sns
    
    colors = sns.color_palette('colorblind')
    
    probs = np.asarray(probs)
    reps = np.asarray(reps)
    
    m = len(probs)
    for i in range(m):
        if ss[i].startswith('www.'):
            ss[i] = ss[i][4:]
    
    plt.figure(figsize=(8,2.0))
    
    gs = gridspec.GridSpec(1, 2, width_ratios=[2.8, 1])
    
    plt.subplot(gs[1]) 
    plt.ylabel('Claim Vera. Prob.')
    plt.bar(range(3), vera, color = colors[3])
    plt.xticks(range(3), ['False', 'Unk.', 'True'])
    
    plt.subplot(gs[0]) 
    plt.xlabel('Source Reputation')
    plt.yticks( np.arange(m) + 0.0, ss)
    
    p1 = plt.barh(range(m), reps * probs[:, 0], color = colors[0], hatch = '//')
    p2 = plt.barh(range(m), reps * probs[:, 1], left = reps * probs[:, 0], color = colors[1])
    p3 = plt.barh(range(m), reps * probs[:, 2], left = reps * probs[:, 0] \
                  + reps * probs[:, 1], color = colors[2], hatch = '\\\\')
    
    
    plt.legend((p1[0], p2[0], p3[0]), ('Against', 'Observing', 'For'), \
               bbox_to_anchor=(0.2, 1.001, 0.5, 0.01), ncol = 3)
    
    plt.savefig(save_name, bbox_inches = 'tight', pad_inches = 0.001, dpi = 300)
        
        