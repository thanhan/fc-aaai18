#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import sys
import os
#sys.path.append(os.path.join('/export/home/u14/atn/research/fact/ferreira'))

sys.path.append(os.path.join('ferreira'))

from model.utils import get_dataset, split_data, RunCV, run_test
from model.classifiers.lr_predictors import LogitPredictor

from model.baseline.transforms import (
    RefutingWordsTransform,
    QuestionMarkTransform,
    HedgingWordsTransform,
    InteractionTransform,
    NegationOfRefutingWordsTransform,
    BoWTransform,
    PolarityTransform,
    BrownClusterPairTransform
)

from model.ext.transforms import (
    AlignedPPDBSemanticTransform,
    NegationAlignmentTransform,
    Word2VecSimilaritySemanticTransform,
    DependencyRootDepthTransform,
    SVOTransform
)

transforms = {
        'BoW': lambda: BoWTransform(),
        'Q': QuestionMarkTransform,
        'W2V': Word2VecSimilaritySemanticTransform,
        'PPDB': AlignedPPDBSemanticTransform,
        'NegAlgn': NegationAlignmentTransform,
        'RootDep': DependencyRootDepthTransform,
        'SVO': SVOTransform,
    }


list_tranforms = [lambda: BoWTransform(),                   # 500
                  QuestionMarkTransform,                    # 1
                  Word2VecSimilaritySemanticTransform,      # 1
                  AlignedPPDBSemanticTransform,             # 1
                  NegationAlignmentTransform,               # 1
                  DependencyRootDepthTransform,             # 2
                  SVOTransform]                             # 12

predictor = LogitPredictor

#p = predictor(transforms.values() )
p = predictor(list_tranforms )


def get_data():
    train_data = get_dataset('url-versions-2015-06-14-clean-train.csv')
    X, y = split_data(train_data)
    X = p.pipeline.fit_transform(X)
    
    train_data1 = train_data[:1489]
    train_data2 = train_data[1489:]
    X1 = X[:1489]
    X2 = X[1489:]
    
    
    test_data = get_dataset('url-versions-2015-06-14-clean-test.csv')
    X_test, y_test = split_data(test_data)
    X_test = p.pipeline.transform(X_test)
    
    # return train/ validation/ test
    return (train_data1, X1, train_data2, X2, test_data, X_test)
    

def get_snopes():
    test_data = get_dataset("my_claims_csv_cleaned.csv")
    X_test, y_test = split_data(test_data)
    X_test = p.pipeline.transform(X_test)
    
    return(test_data, X_test)