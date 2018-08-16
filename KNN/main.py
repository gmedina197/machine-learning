__author__ = "Gabriel Medina, Victor Pozzan"

import pandas as pd
import numpy as np
import KNN
from sklearn.model_selection import KFold


k_value = 3
df = pd.read_csv('Diabetes.csv')
x, y = np.split(df, [len(df.columns) - 1], axis=1)
media = []

for j in range(0, 10):
    kf = KFold(n_splits=5, shuffle=True)
    folds = kf.split(x)
    fold_result = []
    for fold in folds:
        X_train, X_test = x.iloc[fold[0]], x.iloc[fold[1]]
        Y_train, Y_test = y.iloc[fold[0]], y.iloc[fold[1]]
        classifier = KNN.KNN(X_train, Y_train, X_test, Y_test, k_value, "sd", "mv")
        classifier.train()
        fold_result.append(classifier.predict())
    print("Execucao %s" % j)
    print(np.sum(fold_result) / 5)
    media.append(np.sum(fold_result) / 5)
print("----")
print("Resultado medio: %s" % ((np.sum(media) / 10)*100))
