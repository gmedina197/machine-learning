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

    def compute_votes(self, candidates):
        predictions = []
        for votes in candidates:
            votes_count = {}
            for vote in votes:
                v = self.y_train.iloc[vote[1]][0]
                if v not in votes_count:
                    if self.vote_type == "mv":
                        votes_count[v] = 1
                    elif self.vote_type == "pv":
                        votes_count[v] = 1 / vote[0]
                else:
                    if self.vote_type == "mv":
                        votes_count[v] += 1
                    elif self.vote_type == "pv":
                        votes_count[v] += 1 / vote[0]
            predictions.append(sorted(votes_count.items())[0])
        return predictions

    def predict(self):
        candidates = []

        for test_index, test_row in self.x_test.iterrows():
            test = np.array(test_row)
            distances = []
            final_distance = []

            for train_index, train_row in self.x_train.iterrows():
                dist = np.linalg.norm(test - np.array(train_row))
                distances.append(dist)

            distances = np.array(distances)

            for index, d in enumerate(distances):
                if self.distance_type == "ied":
                    final_distance.append((1 / d, index))

                elif self.distance_type == "sd":
                    distances = (distances - np.min(distances)) / (np.max(distances) - np.min(distances))
                    final_distance.append((1 - d, index))

            candidates.append(sorted(final_distance)[:self.k])

        predictions = self.compute_votes(candidates)

        sum_predicions = 0

        for idx, predicted in enumerate(predictions):
            expected = self.y_test.iloc[idx][0]
            sum_predicions += expected == predicted[0]

        return sum_predicions / len(self.x_test)