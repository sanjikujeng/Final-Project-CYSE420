import math  # Add this import at the top of your file

# Naive Bayes Implementation from Scratch
class NaiveBayes:
    def __init__(self):
        self.summaries = {}

    def summarize(self, dataset):
        summaries = [(sum(column) / len(column), sum((x - sum(column) / len(column)) ** 2 for x in column) / len(column))
                     for column in zip(*dataset)]
        return summaries[:-1]

    def fit(self, X, y):
        dataset = [X[i] + [y[i]] for i in range(len(X))]
        separated = {label: [row for row in dataset if row[-1] == label] for label in set(y)}
        self.summaries = {label: self.summarize(rows) for label, rows in separated.items()}

    def calculate_probability(self, x, mean, variance):
        exponent = math.exp(-((x - mean) ** 2 / (2 * variance)))
        return (1 / (math.sqrt(2 * math.pi * variance))) * exponent

    def predict(self, X):
        predictions = []
        for row in X:
            probabilities = {label: 1 for label in self.summaries}
            for label, summaries in self.summaries.items():
                for i in range(len(summaries)):
                    mean, variance = summaries[i]
                    probabilities[label] *= self.calculate_probability(row[i], mean, variance)
            predictions.append(max(probabilities, key=probabilities.get))
        return predictions
