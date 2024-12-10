# Load the dataset manually
def load_data(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    
    data = []
    for line in lines:
        # Split by commas and strip newline characters
        row = line.strip().split(',')
        # Convert all attributes to float except the last one (label)
        row = [float(i) if i != '?' else 0.0 for i in row[:-1]] + [row[-1]]  # Convert label to string
        data.append(row)
    return data

# Split dataset into train and test (80% train, 20% test)
def train_test_split(data, train_size=0.8):
    train_len = int(len(data) * train_size)
    train_data = data[:train_len]
    test_data = data[train_len:]
    
    return train_data, test_data

# Naive Bayes classifier (Gaussian Naive Bayes from scratch)
def naive_bayes(train_X, train_y, test_X):
    class_summaries = {}
    
    # Calculate mean and variance for each feature for each class
    for i in range(len(train_X)):
        class_label = train_y[i]
        if class_label not in class_summaries:
            class_summaries[class_label] = [[], []]  # Two lists for features

        for j in range(len(train_X[i])):
            class_summaries[class_label][j].append(train_X[i][j])

    # Calculate mean and variance
    for class_label, features in class_summaries.items():
        for i in range(len(features)):
            mean = sum(features[i]) / len(features[i])
            variance = sum([(x - mean) ** 2 for x in features[i]]) / len(features[i])
            class_summaries[class_label][i] = (mean, variance)

    # Make predictions
    predictions = []
    for row in test_X:
        probabilities = {}
        for class_label, features in class_summaries.items():
            probabilities[class_label] = 1
            for i in range(len(row)):
                mean, variance = features[i]
                exponent = (-(row[i] - mean) ** 2) / (2 * variance)
                probability = (1 / (variance ** 0.5 * (2 * 3.1415) ** 0.5)) * (2.718 ** exponent)
                probabilities[class_label] *= probability

        prediction = max(probabilities, key=probabilities.get)
        predictions.append(prediction)

    return predictions

# Decision Tree classifier (Single Split)
def decision_tree_classifier(train_X, train_y, test_X):
    def entropy(y):
        class_counts = {}
        for label in y:
            if label not in class_counts:
                class_counts[label] = 0
            class_counts[label] += 1
        entropy_value = 0
        for count in class_counts.values():
            prob = count / len(y)
            entropy_value -= prob * math.log2(prob)
        return entropy_value

    def best_split(X, y):
        best_entropy = float('inf')
        best_split = None
        best_feature = None
        for feature_index in range(len(X[0])):
            feature_values = [x[feature_index] for x in X]
            unique_values = set(feature_values)
            for value in unique_values:
                left_split_y = [y[i] for i in range(len(y)) if X[i][feature_index] == value]
                right_split_y = [y[i] for i in range(len(y)) if X[i][feature_index] != value]
                entropy_value = (len(left_split_y) / len(y)) * entropy(left_split_y) + \
                                 (len(right_split_y) / len(y)) * entropy(right_split_y)
                if entropy_value < best_entropy:
                    best_entropy = entropy_value
                    best_split = value
                    best_feature = feature_index
        return best_split, best_feature

    # Build a simple decision tree (one split)
    split_value, feature_index = best_split(train_X, train_y)

    predictions = []
    for row in test_X:
        if row[feature_index] == split_value:
            predictions.append(1)  # Class 1
        else:
            predictions.append(0)  # Class 0

    return predictions

# Evaluate predictions (accuracy and F1-score)
def evaluate_predictions(predictions, true_labels):
    correct = sum([1 for p, t in zip(predictions, true_labels) if p == t])
    accuracy = correct / len(true_labels) * 100

    # F1-score calculation (binary case)
    tp = sum([1 for p, t in zip(predictions, true_labels) if p == 1 and t == 1])
    fp = sum([1 for p, t in zip(predictions, true_labels) if p == 1 and t == 0])
    fn = sum([1 for p, t in zip(predictions, true_labels) if p == 0 and t == 1])
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    return accuracy, f1_score

# Main function to run everything
if __name__ == '__main__':
    # Load the KDD data
    train_data = load_data('data/kddtrain+.data')
    test_data = load_data('data/kddtest+.data')

    # Split into features and labels
    train_X = [row[:-1] for row in train_data]
    train_y = [row[-1] for row in train_data]
    test_X = [row[:-1] for row in test_data]
    test_y = [row[-1] for row in test_data]

    # Train-test split for validation
    # (Uncomment if you want to perform train-test splitting manually)
    # train_data, test_data = train_test_split(data)
    # train_X = [row[:-1] for row in train_data]
    # train_y = [row[-1] for row in train_data]
    # test_X = [row[:-1] for row in test_data]
    # test_y = [row[-1] for row in test_data]

    # Test Simple Classifier (threshold-based)
    simple_predictions = [1 if sum(features) > 8 else 0 for features in test_X]
    simple_accuracy, simple_f1 = evaluate_predictions(simple_predictions, test_y)
    print(f"Simple Classifier -> Accuracy: {simple_accuracy:.2f}%, F1-Score: {simple_f1:.2f}")

    # Test Naive Bayes
    nb_predictions = naive_bayes(train_X, train_y, test_X)
    nb_accuracy, nb_f1 = evaluate_predictions(nb_predictions, test_y)
    print(f"Naive Bayes -> Accuracy: {nb_accuracy:.2f}%, F1-Score: {nb_f1:.2f}")

    # Test Decision Tree Classifier
    dt_predictions = decision_tree_classifier(train_X, train_y, test_X)
    dt_accuracy, dt_f1 = evaluate_predictions(dt_predictions, test_y)
    print(f"Decision Tree -> Accuracy: {dt_accuracy:.2f}%, F1-Score: {dt_f1:.2f}")
