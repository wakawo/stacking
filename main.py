
# coding: UTF-8

import numpy as np

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.naive_bayes import GaussianNB as GNB
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.ensemble import BaggingClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support

from stacking import StackingClassifier

def main():
    digits = load_digits()
    noised_data = digits.data + np.random.random(digits.data.shape)*15

    X_train, X_test, y_train, y_test = train_test_split(
        noised_data, digits.target, test_size=0.8)

    svm =SVC(C=5, gamma=0.001, probability=True)
    lr = LogisticRegression()
    knn = KNN(n_jobs=-1)
    nb = GNB()
    rfc = RFC(n_estimators=500, n_jobs=-1)
    bgg = BaggingClassifier(n_estimators=300, n_jobs=-1)
    mlp = MLPClassifier(hidden_layer_sizes=(40, 20), max_iter=1000)
    xgb = XGBClassifier(n_estimators=300, n_jobs=-1)

    estimators = list(zip(["svm","lr","knn","nb","rfc","bgg","mlp","xgb"],
                          [svm,lr,knn,nb,rfc,bgg,mlp,xgb]))
    
    for name, clf in estimators:
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        print(name)
        print("p:{0:.4f} r:{1:.4f} f1:{2:.4f}".format(
            *precision_recall_fscore_support(y_test, preds, average="macro")))

    for v in ["hard", "soft"]:
        vc_hard = VotingClassifier(estimators, voting=v)
        vc_hard.fit(X_train, y_train)
        preds = vc_hard.predict(X_test)
        print(v, "voting")
        print("p:{0:.4f} r:{1:.4f} f1:{2:.4f}".format(
            *precision_recall_fscore_support(y_test, preds, average="macro")))

    stcl = StackingClassifier(estimators, RFC(n_estimators=2000, n_jobs=-1))
    stcl.fit(X_train, y_train)
    preds = stcl.predict(X_test)
    print("stacking")
    print("p:{0:.4f} r:{1:.4f} f1:{2:.4f}".format(
        *precision_recall_fscore_support(y_test, preds, average="macro")))
    
if __name__ == "__main__":
    main()
