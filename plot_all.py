#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May  9 16:33:23 2017

@author: thanhan
"""



import active
import pickle

#res = pickle.load(open('save_gibbs_2k_seed_10_20.pkl'))
res1 = pickle.load(open('save_a2'))
res = pickle.load(open('save_a2.pkl'))
res_bl = pickle.load(open('save_res_bl.pkl') )

active.plot_curves([res[0][:, :, 0].T, res[1][:, :, 0].T, res[2][:, :, 0].T], ['Baseline', 'Variational', 'Gibbs'], time = range(0, 2001, 400), xlab = 'Stance labels by journalists', ylab = 'Stance accuracy', save_name='fact_stance.pdf')
#active.plot_curves([res[0][:, :, 1].T, res[1][:, :, 1].T, res[2][:, :, 1].T], ['Baseline', 'Variational', 'Gibbs'], time = range(0, 2001, 400), xlab = 'Stance labels by journalists', ylab = 'Veracity accuracy', save_name='fact_vera.pdf')
active.plot_curves([res_bl[:, :, 2].T, res[1][:, :, 2].T, res[2][:, :, 2].T], ['Baseline', 'Variational', 'Gibbs'], time = range(0, 2001, 400), xlab = 'Stance labels by journalists', ylab = 'Brier score', save_name='fact_brier.pdf')


active.plot_curves([res[0][:, :, 2].T, res[1][:, :, 2].T, res[2][:, :, 2].T], ['Baseline', 'Variational', 'Gibbs'], time = range(0, 2001, 400), xlab = 'Stance labels by journalists', ylab = 'Brier score', save_name='fact_brier.pdf')

eres1 = pickle.load(open('save_e3.pkl'))
eres1.update(pickle.load(open('save_e4.pkl')))
active.plot_curves([active.take_res(eres1[(1,10,1,0)], 1), active.take_res(eres1[(2,10,1,0)], 1)], ['Baseline', 'Variational'], time = range(0, 500, 50), xlab = 'Test-set stance labels by crowds', ylab = 'Veracity accuracy', save_name = 'active_vera.pdf')
active.plot_curves([active.take_res(eres1[(1,10,1,0)], 0), active.take_res(eres1[(2,10,1,0)], 0)], ['Baseline', 'Variational'], time = range(0, 500, 50), xlab = 'Test-set stance labels by crowds', ylab = 'Stance accuracy', save_name='active_stance.pdf')
active.plot_curves([active.take_res(eres1[(1,10,1,0)], 2), active.take_res(eres1[(2,10,1,0)], 2)], ['Baseline', 'Variational'], time = range(0, 500, 50), xlab = 'Test-set stance labels by crowds', ylab = 'Brier score', save_name='active_brier.pdf')


active.plot_curves([active.take_res(eres1[(1,10,1,0)], 1)])

active.plot_curves([active.take_res(eres1[(2,10,1,0)], 2), active.take_res(eres1[(2,0,0,0)], 2)], ['10-1-1', '0-0-0'], time = range(0, 500, 50), xlab = 'Test-set stance labels by crowds', ylab = 'Veracity accuracy')


active.plot_curves([active.take_res(eres1[(1,10,1,0)], 2), active.take_res(eres1[(2,10,1,0)], 2)], ['Baseline', 'Variational'], time = range(0, 500, 50), xlab = 'Test-set stance labels by crowds', ylab = 'Brier score', save_name='active_brier.pdf')


active.plot_curves([res[0][:, :, 2].T, res[1][:, :, 2].T], ['Baseline', 'Variational'], time = range(0, 2001, 400), xlab = 'Stance labels by journalists', ylab = 'Stance accuracy')
active.plot_curves([res[0][:, :, 0].T, res[1][:, :, 0].T], ['Baseline', 'Variational'], time = range(0, 2001, 400), xlab = 'Stance labels by journalists', ylab = 'Stance accuracy')


active.plot_curves([active.take_res(eres1[(1,0,0,0)], 2), active.take_res(eres1[(2,0,0,0)], 2)], ['Baseline', 'Variational'], time = range(0, 500, 50), xlab = 'Test-set stance labels by crowds', ylab = '')


import models
dic_slug = models.get_claim_slug()
cmv = models.model_cv(data_all, X, cds, cdv)
cmv.init_model()
cmv.em(5)
(cs, probs, vera) = models.get_source_prob(cmv, 269, dic_slug)
(cs, probs, vera) = models.take_subset(cs, probs, vera, [0, 4, 6, 7, 14, 16])
models.plot_source(cs, probs, vera, save_name='ex2.pdf')

(probs, reps, ss, vera) = cmv.get_prob()
models.plot_probs(probs, reps, ss, vera, save_name='example.pdf')

############################################
eres1 = pickle.load(open('save_e3.pkl'))
eres1.update(pickle.load(open('save_e4.pkl')))

eres1.update(pickle.load(open('active_gibbs3.pkl')))
eres1.update(pickle.load(open('active_cmv3.pkl')))
# active includes gibbs
active.plot_curves([active.take_res(eres1[(1,10,1,0)], 2), active.take_res(eres1[(2,10,1,0)], 2), active.take_res(eres1[(3,10,1,0)], 2)], ['Baseline', 'Variational', 'Gibbs'], time = range(0, 500, 50), xlab = 'Test-set stance labels by crowds', ylab = 'Brier score', save_name='active_brier.pdf')


active.plot_curves([active.take_res(eres1[(1,10,1,0)], 0), active.take_res(eres1[(2,10,1,0)], 0), active.take_res(eres1[(3,10,1,0)], 0)], ['Baseline', 'Variational', 'Gibbs'], time = range(0, 500, 50), xlab = 'Test-set stance labels by crowds', ylab = 'Brier score', save_name='active_stance.pdf')


def plot_survey():
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    colors = sns.color_palette('colorblind')
    a = [1,6,3,1,0]
    b = [0,4,3,5,0]
    
    plt.yticks( range(5), ['Very Dis', 'Dis', 'Neu', 'Very S', 'Very sa'])
    
    bins = range(5)
    
    p1 = plt.hist(np.vstack([a,b]).T, bins, color=colors[3:5], label = ['Only prediction', 'With explanation'], orientation='horizontal')
    
    plt.legend()
    
    
def plot_survey2():

    
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    colors = sns.color_palette('colorblind')
    a = [1,6,3,1,0]
    b = [0,4,3,5,0]
    
    
    plt.yticks( range(5), ['Very Disatisfied', 'Dissatisfied', 'Neutral', 'Satisfied', 'Very satisfied'])
    
    plt.xlabel('Number of users')
    
    p1 = plt.barh(np.arange(5)-0.15 , a, 0.30, color = colors[0],hatch = 'x')
    p2 = plt.barh(np.arange(5)+0.15 , b, 0.30, color = colors[1])    
    
    plt.legend((p1[0], p2[0]), ('Group 1: Only prediction', 'Group 2: With explanation'))
    
    plt.savefig('survey.pdf', bbox_inches = 'tight', pad_inches = 0.001, dpi = 300)
    
    

active.plot_curves([active.take_res(eres1[(1,10,1,0)], 0), active.take_res(eres1[(2,10,1,0)], 0), active.take_res(eres1[(3,10,1,0)], 0)], ['Baseline', 'Variational', 'Gibbs'], time = range(0, 500, 50), xlab = 'Test-set stance labels by crowds', ylab = 'Stance accuracy', save_name='active_stance.pdf')
