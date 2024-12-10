# K-Nearest Neighbors Implementation from Scratch
class KNN:
    def __init__(self, k=3):
        self.k = k

    def euclidean_distance(self, row1, row2):
        return sum((row1[i] - row2[i]) ** 2 for i in range(len(row1))) ** 0.5

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = []
        for row in X:
            distances = [(self.euclidean_distance(row, self.X_train[i]), self.y_train[i]) for i in range(len(self.X_train))]
            distances.sort(key=lambda x: x[0])
            neighbors = [distances[i][1] for i in range(self.k)]
            predictions.append(max(set(neighbors), key=neighbors.count))
        return predictions
