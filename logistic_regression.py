# Logistic Regression Implementation from Scratch
import math

class LogisticRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = []
        self.bias = 0

    def sigmoid(self, z):
        return 1 / (1 + math.exp(-z))

    def fit(self, X, y):
        self.weights = [0] * len(X[0])  # Initialize weights
        self.bias = 0

        for _ in range(self.epochs):
            for i in range(len(X)):
                z = sum([X[i][j] * self.weights[j] for j in range(len(self.weights))]) + self.bias
                prediction = self.sigmoid(z)
                error = y[i] - prediction

                # Update weights and bias
                for j in range(len(self.weights)):
                    self.weights[j] += self.learning_rate * error * X[i][j]
                self.bias += self.learning_rate * error

    def predict(self, X):
        predictions = []
        for i in range(len(X)):
            z = sum([X[i][j] * self.weights[j] for j in range(len(self.weights))]) + self.bias
            predictions.append(1 if self.sigmoid(z) >= 0.5 else 0)
        return predictions
