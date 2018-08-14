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
X, Y = get_data()
media_result = []

for j in range(0, 10):
    kfolds = get_folds(X)
    fold_result = []
    for fold in kfolds:
        X_train, X_test = X.iloc[fold[0]], X.iloc[fold[1]]
        Y_train, Y_test = Y.iloc[fold[0]], Y.iloc[fold[1]]
        classifier = KNN.KNN()
        classifier.train(X_train, Y_train)
        fold_result.append(classifier.predict(X_test, Y_test, k_value, distance_type="st"))
    print(fold_result)
    #print(np.sum(fold_result) / 5)
    #media_result.append(np.sum(fold_result) / 5)
print("Media folds")
print(np.sum(media_result) / 10)