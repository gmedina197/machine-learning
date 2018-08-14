import numpy as np
from collections import Counter

class KNN:

    def train(self, X, Y):
        self.X = X
        self.Y = Y

    def standard_distance(self, distances):
        return (distances - np.min(distances)) / (np.max(distances) - np.min(distances))

    def predict(self, X_test, Y_test, k, distance_type):

        for test_index, test_row in X_test.iterrows():
            test = np.array(test_row)
            distances = []

            for train_index, train_row in self.X.iterrows():
                train = np.array(train_row)
                distance = np.linalg.norm(test - train)
                distances.append(distance)

            distances = np.array(distances)
            if distance_type == "st":
                distances = self.standard_distance(np.array(distances))
            # inverso da dist euclidiana

            # distancia normalizada