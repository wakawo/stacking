
# coding: utf-8

# In[15]:


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
            clf_origin = self.original_clfs[clf_name]
            preds_tmp_list = []
            for train_index, test_index in index_list:
                clf_copy = deepcopy(clf_origin)
                clf_copy.fit(X[train_index], y[train_index])
                preds_tmp_list.append(
                    clf_copy.predict_proba(X[test_index]))
                self.clfs_dict[clf_name].append(clf_copy)
            merge_feature_list.append(np.vstack(preds_tmp_list))
        
        X_merged = np.hstack(merge_feature_list)
        y_merged = np.hstack([y[test_index] 
                              for _, test_index in index_list])

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


# In[1]:


# coding: UTF-8

import numpy as np

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.naive_bayes import GaussianNB as GNB
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.ensemble import BaggingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support

digits = load_digits()
noised_data = digits.data + np.random.random(digits.data.shape)*15

X_train, X_test, y_train, y_test = train_test_split(noised_data, digits.target, test_size=0.8)

svm =SVC(C=5, gamma=0.001, probability=True)
lr = LogisticRegression()
knn = KNN(n_jobs=-1)
nb = GNB()
rfc = RFC(n_estimators=500, n_jobs=-1)
bgg = BaggingClassifier(n_estimators=300, n_jobs=-1)
mlp = MLPClassifier(hidden_layer_sizes=(40, 20), max_iter=1000)



# In[21]:


lr1 = LogisticRegression()
lr2 = LogisticRegression()
lr3 = LogisticRegression()
lr4 = LogisticRegression()


# In[22]:


estimators = list(zip(["lr_1", "lr_2", "lr_3", "lr_4"],
                      [lr1, lr2, lr3, lr4]))


# In[24]:


stcl = StackingClassifier(estimators, LogisticRegression())
stcl.fit(X_train, y_train)
preds = stcl.predict(X_test)
print("stacking")
print("p:{0:.4f} r:{1:.4f} f1:{2:.4f}".format(
    *precision_recall_fscore_support(y_test, preds, average="macro")))

