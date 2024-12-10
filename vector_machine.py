class SVM:
    def __init__(self, learning_rate=0.01, epochs=1000, lambda_param=0.1):
        """
        SVM classifier using stochastic gradient descent for training.

        :param learning_rate: Learning rate for the SGD algorithm.
        :param epochs: Number of iterations to train the model.
        :param lambda_param: Regularization parameter.
        """
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.lambda_param = lambda_param
        self.weights = None  # weights vector
        self.bias = 0  # bias term

    def fit(self, X, y):
        """
        Train the SVM classifier using the training data.

        :param X: Training data (list of lists, each inner list is a sample).
        :param y: Target labels (list of values, with values -1 or 1).
        """
        # Initialize weights as a list of zeros
        self.weights = [0] * len(X[0])  # Number of features

        # Training loop
        for epoch in range(self.epochs):
            for i in range(len(X)):
                # Calculate the decision function: y_i * (w * x_i + b)
                dot_product = sum([X[i][j] * self.weights[j] for j in range(len(X[i]))]) + self.bias
                condition = y[i] * dot_product >= 1

                if condition:
                    # Regularization part (only update weights)
                    self.weights = [w - self.learning_rate * (2 * self.lambda_param * w) for w in self.weights]
                else:
                    # Update weights and bias
                    self.weights = [self.weights[j] - self.learning_rate * (2 * self.lambda_param * self.weights[j] - X[i][j] * y[i]) for j in range(len(self.weights))]
                    self.bias -= self.learning_rate * y[i]

    def predict(self, X):
        """
        Predict labels for the given samples.

        :param X: Samples to predict (list of lists).
        :return: Predicted labels (list).
        """
        predictions = []
        for i in range(len(X)):
            # Linear prediction rule
            prediction = sum([X[i][j] * self.weights[j] for j in range(len(X[i]))]) + self.bias
            predictions.append(1 if prediction >= 0 else -1)
        return predictions
