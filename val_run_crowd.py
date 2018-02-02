# for VALIDATION
import crowd
import models
import pickle
import numpy as np
import pandas as pd
import util

(train_data_pp, X_train, val_data_pp, X_val, test_data_pp, X_test) = \
pickle.load( open('edata_pp.pkl') )

m = models.model(train_data_pp, X_train, val_data_pp, X_val, seed = 1, \
                 sample_its=2000, burn = 1000)
m.init_model()

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



data = pd.concat((data1, data2, data3, data4, data5, data6, data7,\
                  data8, data9, data10, data11, data12, data13))

erange = range(1490, 2072)
#erange = []
(data_all, X, cds, cdv, cds_test) = models.prepare_cm_data(train_data_pp, X_train, \
                        val_data_pp, X_val, data, expert_range = erange)

#cm = models.crowd_model(m.data_all, m.X, cds, cdv)
#cm.init_model()

#cmv = models.model_cv(m.data_all, m.X, cds, cdv)
#cmv.init_model()
