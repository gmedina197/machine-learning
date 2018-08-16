__author__ = "Gabriel Medina, Victor Pozzan"

import pandas as pd
import numpy as np
import KNN
from sklearn.model_selection import KFold


def get_data():
    df = pd.read_csv('Diabetes.csv')
    return np.split(df, [len(df.columns) - 1], axis=1)


def get_folds(dataset):
    kf = KFold(n_splits=5, shuffle=True)
    return kf.split(dataset)


k_value = 3
x, y = get_data()
media_result = []

for j in range(0, 10):
    kfolds = get_folds(x)
    fold_result = []
    for fold in kfolds:
        X_train, X_test = x.iloc[fold[0]], x.iloc[fold[1]]
        Y_train, Y_test = y.iloc[fold[0]], y.iloc[fold[1]]
        classifier = KNN.KNN(X_train, Y_train, X_test, Y_test, k_value, distance_type="sd", vote_type="mv")
        classifier.train()
        fold_result.append(classifier.predict())
    print("Execucao %s" % j)
    print(np.sum(fold_result) / 5)
    media_result.append(np.sum(fold_result) / 5)
print("Resultado medio: %s" % ((np.sum(media_result) / 10)*100))
print(np.sum(media_result) / 10)