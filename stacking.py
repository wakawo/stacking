
# coding: UTF-8

from copy import deepcopy
from collections import defaultdict

import numpy as np
from sklearn.model_selection import StratifiedKFold

class StackingClassifier:
    def __init__(self, estimators, marge_estimator):
        self.original_clfs = dict(estimators)
        self.m_clf = marge_estimator

        self.clfs_dict = defaultdict(list)
        self.clfs_index = sorted(self.original_clfs.keys())

    def fit(self, X, y):
        skf = StratifiedKFold(n_splits=15)
        index_list = list(skf.split(X, y))

        merge_feature_list = []
        for clf_name in self.clfs_index:
            print("Classifier {} start.".format(clf_name))
            clf_origin = self.original_clfs[clf_name]
            preds_tmp_list = []
            for i, (train_index, test_index) in enumerate(index_list):
                print(i, end=' ')
                clf_copy = deepcopy(clf_origin)
                clf_copy.fit(X[train_index], y[train_index])
                preds_tmp_list.append(
                    clf_copy.predict_proba(X[test_index]))
                self.clfs_dict[clf_name].append(clf_copy)
            
            print()
            merge_feature_list.append(np.vstack(preds_tmp_list))
        
        X_merged = np.hstack(merge_feature_list)
        y_merged = np.hstack([y[test_index] 
                              for _, test_index in index_list])
        
        print("merge classifier start.")
        self.m_clf.fit(X_merged, y_merged)
    
    def predict(self, X):
        merge_feature_list = []
        for clf_name in self.clfs_index:
            tmp_proba_list = []
            for clf in self.clfs_dict[clf_name]:
                tmp_proba_list.append(clf.predict_proba(X))
            merge_feature_list.append(
                np.mean(tmp_proba_list, axis=0))
        X_merged = np.hstack(merge_feature_list)

        return self.m_clf.predict(X_merged)
    
    def predict_proba(self, X):
        merge_feature_list = []
        for clf_name in self.clfs_index:
            tmp_proba_list = []
            for clf in self.clfs_dict[clf_name]:
                tmp_proba_list.append(clf.predict_proba(X))
            merge_feature_list.append(
                np.mean(tmp_proba_list, axis=0))
        X_merged = np.hstack(merge_feature_list)

        return self.m_clf.predict_proba(X_merged)
