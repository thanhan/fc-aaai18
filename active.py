
import crowd
import util
import models

import scipy
import numpy as np

import pandas as pd

class sim:
    """
    simulate active label collection
    """
    
    def __init__(self, train_data, X_train, test_data, X_test, cds, cdv, \
                 cds_test, k = 50):
        """
        cd = crowd data
        train_data: have some labels (crowd or expert)
        test_data: have no labels
        """
        self.train_data = train_data
        self.X_train = X_train
        self.test_data = test_data
        self.X_test = X_test
        self.cds = cds
        self.cdv = cdv
        self.cds_test = cds_test
        self.k = k
        self.selected = np.zeros((len(test_data,)))
        self.res = []
        
        
    def get_aids(self, indices):
        """
        return aids
        exclude selected
        """
        res = []
        for i in range(len(indices)):
            j = indices[i]
            if self.selected[j] == 1: continue # exclude selected
            res.append(self.test_data.articleCount.iloc[j])
        
        return res
        
    def query(self, indices):
        """
        indices = indices of articles in test_data to get crowd labels
        """

        # keep train_data and test_data, just move crowd labels
        # from cds_test to cds
        
        
        
        aids = self.get_aids(indices) # aids of selected articles
        self.selected[indices] = 1
        
        f1 = map(lambda x: x in aids, self.cds_test.data.aid)
        #f0 = map(lambda x: x not in aids, self.cds_test.data.aid)
        
        new_cds_data      = pd.concat( [self.cds.data.copy(), \
                                        self.cds_test.data.iloc[f1]])
        #new_cds_test_data = self.cds_test.data.iloc[f0]
        
        self.cds = crowd.CD(new_cds_data)
        #self.cds_test = crowd.CD(new_cds_test_data)

    def run(self, method, n_run = 10):
        method.k = self.k
        
        for i in range(n_run):
            (ps, pt, pp_t, ids) = method.run()
            r = util.get_acc(self.test_data, ps, pt, pp_t)
            self.res.append(r)
            
            print i, 'stance_acc, claim_acc, brier_score=', r
            
            self.query(ids)
            method.update_cds(self.cds)
    
    
###############################################################################
    
class active_selector:
    """
    select article to get stance labels from crowd
    """

    def __init__(self, train_data, X_train, test_data, X_test, cds, cdv, k = 50\
                 , seed = 1):
        """
        """
        self.train_data = train_data
        self.X_train = X_train
        self.test_data = test_data
        self.X_test = X_test
        self.cds = cds
        self.cdv = cdv
        self.k = k
        self.n_test = len(test_data)
        
        # flag for selected test articles
        self.selected = np.zeros((len(test_data)))
        self.rs = np.random.RandomState(seed = seed)
        
        self.n_train = len(train_data)
        self.n_train_vera = len(train_data.claimCount.unique())
    
    def eval_score_uncer(self, pp_s):
        """
        eval. scores by uncertainty sampling
        """
        self.scores = []
        for i in range(pp_s.shape[0]):
            self.scores.append(scipy.stats.entropy(pp_s[i, :]))
    
    def run_method(self):
        """
        run the method to predict stance/vera and calculate scores
        """
        (self.ps, self.pt, self.pp_s, self.pp_t) = util.baseline_crowd(self.train_data,\
        self.X_train, self.test_data, self.X_test, self.cds, self.cdv, True)
        
        self.eval_score_uncer(pp_s)
    
    def run(self):
        """
        return predicted stance/vera and selection scores
        """

        self.run_method()            
        self.scores = self.scores * (1 - self.selected)
        #self.sorted_score_ids = np.argsort(self.scores)[::-1] # reverse array
        #self.selected[self.sorted_score_ids[:self.k]] = 1
        
        self.sl_ids = self.rs.choice(range(self.n_test), size = self.k, replace = False, \
                       p = util.softmax(self.scores))
        self.selected[self.sl_ids] = 1
        
        return (self.ps, self.pt, self.pp_t, self.sl_ids)
        
    def update_cds(self, cds):
        self.cds = cds
        
    def cal_scores(self, clf_vera):
        """
        score = hc * claim entropy + hr * Reputation + hs * stance entropy
        """
        test_mat = self.test_data[['claimCount', 'articleCount',\
                                   'sourceCount']].as_matrix()

        self.scores = []
        list_claim_en = []
        list_rep = []
        list_stance_en = []
        
        for i in range(len(test_mat)):
            stance_en = scipy.stats.entropy(self.pp_s[i, :])
            source = test_mat[i][2] - 1
            rep = clf_vera.coef_[:, source]
            rep = np.sum(np.abs(rep))
            claimCount = test_mat[i][0] - self.n_train_vera - 1
            claim_en = scipy.stats.entropy(self.pp_t[claimCount, :])
            
            list_claim_en.append(claim_en)
            list_rep.append(rep)
            list_stance_en.append(stance_en)
        
        # normalize
        list_claim_en  = np.asarray(list_claim_en)  / np.sum(list_claim_en)
        list_rep       = np.asarray(list_rep)       / np.sum(list_rep)
        list_stance_en = np.asarray(list_stance_en) / np.sum(list_stance_en)
        
        # calcuate scores
        for i in range(len(test_mat)):
            self.scores.append(self.hc * list_claim_en[i]   + \
                               self.hr * list_rep[i]        + \
                               self.hs * list_stance_en[i])
    

    def set_hp(self, hc, hr, hs):
        """
        set hyper-params
        """
        self.hc = hc
        self.hr = hr
        self.hs = hs

class selector1(active_selector):
    """
    selector baseline
    """
    def run_method(self):
        (self.ps, self.pt, self.pp_s, self.pp_t, clf_st, clf_vera) = \
        util.baseline_crowd(self.train_data, self.X_train, self.test_data,\
                            self.X_test, self.cds, self.cdv, True)
        
        self.cal_scores(clf_vera)


class selector2(active_selector):
    """
    selector by cmv (crowd model w variatinal inference)
        
    """
    

    def run_method(self):
        """
        hc, hr, hs: hyper-params
        n_train      = # train stances
        n_train_vera = # train claims
        """
        
        data_all = pd.concat([self.train_data, self.test_data], \
                             ignore_index = True)
        
        X = scipy.sparse.vstack((self.X_train, self.X_test))
        
        cmv = models.model_cv(data_all, X, self.cds, self.cdv)
        cmv.init_model()
        cmv.em(3)
        
        
        (self.ps, self.pt, self.pp_s, self.pp_t) = cmv.get_res(n_train = self.n_train, \
        n_train_vera = self.n_train_vera)
        
        self.cal_scores(cmv.clf_vera)
        
        
class selector3(active_selector):
    """
    selector by cmv (crowd model w gibbs sampling)
        
    """
    

    def run_method(self):
        """
        hc, hr, hs: hyper-params
        n_train      = # train stances
        n_train_vera = # train claims
        """
        
        data_all = pd.concat([self.train_data, self.test_data], \
                             ignore_index = True)
        
        X = scipy.sparse.vstack((self.X_train, self.X_test))
        
        cm = models.crowd_model(data_all, X, self.cds, self.cdv)
        cm.init_model()
        cm.em(3)
        
        
        (self.ps, self.pt, self.pp_s, self.pp_t) = cm.get_res(n_train = self.n_train, \
        n_train_vera = self.n_train_vera)
        
        self.cal_scores(cm.clf_vera)

        
class experiment:
    """
    class for running experiment
    """
    def __init__(self, train_data, X_train, test_data, X_test, cds, cdv, \
                 cds_test, runs = 10):
        self.train_data = train_data
        self.X_train = X_train
        self.test_data = test_data
        self.X_test = X_test
        self.cds = cds
        self.cdv = cdv
        self.cds_test = cds_test
        
        self.res = {}
        
        self.runs = runs
        
        
        
    def do_exp(self, selector, hc, hr, hs):
        """
        do experiments w a configuration
        save to self.res
        selector 1 = baseline
                 2 = variational
                 3 = gibbs
        """
        self.ss = []
        
        self.save_res = []
        
        self.run(self.train_data, self.X_train, self.test_data, self.X_test, \
                 self.cds, self.cdv, self.cds_test, hc = hc, hr = hr, hs = hs,\
                 selector = selector)
        
        self.res[(selector, hc, hr, hs)] = self.save_res
        
        
    def run(self, train_data, X_train, test_data, X_test, cds, cdv, cds_test,\
                seeds = None, hc = 1, hr = 1, hs = 1, selector = 1):
        """
        """
        
        runs = self.runs
        if seeds == None: seeds = range(runs)
        for seed in seeds:
            S = sim(train_data, X_train, test_data, X_test, cds, cdv, cds_test)
            
            if selector == 1:
                s = selector1(train_data, X_train, test_data, X_test, cds, cdv, seed=seed)
            elif selector == 2:
                s = selector2(train_data, X_train, test_data, X_test, cds, cdv, seed=seed)
            elif selector == 3:
                s = selector3(train_data, X_train, test_data, X_test, cds, cdv, seed=seed)
            else:
                raise "no such selector"
            
            s.set_hp(hc, hr, hs)
            
            S.run(s)
            self.save_res.append(S.res)
            #self.save_sim.append(S)
            #self.save_sel.append(s)
    
    
def take_res(res, p):
    x = []
    for r in res:
        x.append(zip(*r)[p])
    return x
        
def plot_curves(saves, conds = ['Baseline', 'Ours'], time = None, xlab = '', ylab = '',\
                save_name = 'abc.png', xl=None):
    #import matplotlib
    #matplotlib.rcParams['pdf.fonttype'] = 3
    
    import seaborn as sns
    sns.plt.figure(figsize=(8,6))
    sns.plt.xlabel(xlab, fontsize = 18)
    sns.plt.ylabel(ylab, fontsize = 18)
    
    #sns.plt.xlim([0, 4000])
    
    
    #colors = ['red', 'blue', 'black']
    colors = sns.color_palette('colorblind')
    markers = ['s', 'o', '^', 'v', '<']
    if time == None: time = range(len(saves[0][0]))
    for i, save in enumerate(saves):
        g = sns.tsplot(data= save, condition = conds[i], \
                   color = colors[i], marker = markers[i], time=time, markersize = 10)
        if xl:
            g.set(xlim=(0, xl))
        
    sns.plt.legend(loc = 'best', fontsize=16)
    sns.plt.savefig(save_name, bbox_inches = 'tight', pad_inches = 0.1)
    
    
