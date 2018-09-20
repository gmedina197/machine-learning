from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

data_csv = pd.read_csv('database/Diabetes.csv')

KNN = []
DT = []
Bernoulli = []
SVM = []
MLP = []


def accuracy(y_true, y_pred):
    return np.mean(np.equal(y_true, y_pred))


for i in range(10):
    x = data_csv.iloc[:, 0:7].values
    y = data_csv.iloc[:, 8].values

    train_x, another_x, train_y, another_y = train_test_split(x, y, test_size=0.5)
    validation_x, test_x, validation_y, test_y = train_test_split(another_x, another_y, test_size=0.5)

    # KNN
    knn = KNeighborsClassifier(n_neighbors=5, algorithm='auto').fit(train_x, train_y)
    KNN.append(accuracy(test_y, knn.predict(test_x)))

    # Decision Tree
    decision = DecisionTreeClassifier().fit(train_x, train_y)
    DT.append(accuracy(test_y, decision.predict(test_x)))

    # Bernoulli
    bn = BernoulliNB().fit(train_x, train_y)
    Bernoulli.append(accuracy(test_y, bn.predict(test_x)))

    # SVM
    svm = SVC().fit(train_x, train_y)
    SVM.append(accuracy(test_y, svm.predict(test_x)))

    # MLP
    mlp = MLPClassifier(hidden_layer_sizes=8, activation='logistic', batch_size=5, max_iter=1000).fit(train_x, train_y)
    MLP.append(accuracy(test_y, mlp.predict(test_x)))

with open("medias.csv", "w") as fp:

    fp.write("Knn, Dst, Nb, Svm, Mlp\n")
    for index in range(10):
        fp.write("%f, %f, %f, %f, %f\n" % (KNN[index], DT[index], Bernoulli[index], SVM[index], MLP[index]))

    fp.write("%f, %f, %f, %f, %f\n" % (np.mean(KNN), np.mean(DT), np.mean(Bernoulli), np.mean(SVM), np.mean(MLP)))

