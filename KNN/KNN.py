import numpy as np


class KNN:

    def __init__(self, x_train, y_train, x_test, y_test, k, distance_type, vote_type):
        self.y_test = y_test
        self.x_test = x_test
        self.y_train = y_train
        self.x_train = x_train
        self.vote_type = vote_type
        self.distance_type = distance_type
        self.k = k

    # lazy algorithm
    def train(self):
        return

    @staticmethod
    def compute_votes(self, candidates):
        return [0]

    def predict(self):

        for test_index, test_row in self.x_test.iterrows():
            test = np.array(test_row)
            distances = []
            final_distances = []

            for train_index, train_row in self.x_train.iterrows():
                train = np.array(train_row)
                distance = np.linalg.norm(test - train)
                distances.append(distance)

            distances = np.array(distances)
            if self.distance_type == "st":
                distances = (distances - np.min(distances)) / (np.max(distances) - np.min(distances))

            for idx, dist in enumerate(distances):
                if self.distance_type == "st":
                    final_distances.append((1-dist, idx))
                elif self.distance_type == "ied":
                    final_distances.append((1/dist, idx))

            candidates = sorted(final_distances)[:self.k]

            predictions = self.compute_votes(candidates)