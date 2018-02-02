#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 15:25:19 2017

@author: atn
"""

import scipy

scipy.stats.ttest_ind([1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 4], [2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 4])


# 2 4 4 1 0
# 1 3 3 5 0

scipy.stats.ttest_ind([1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4], [1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 4])