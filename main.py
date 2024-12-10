# Load the dataset manually
def load_data(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    
    data = []
    for line in lines:
        row = line.strip().split(',')
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
            entropy_value -= prob * log2(prob)
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

    split_value, feature_index = best_split(train_X, train_y)

    predictions = []
    for row in test_X:
        if row[feature_index] == split_value:
            predictions.append(1)  # Class 1
        else:
            predictions.append(0)  # Class 0

    return predictions

# Random Forest Classifier (using multiple decision trees)
def random_forest(train_X, train_y, test_X, num_trees=5):
    def bootstrap_sample(X, y):
        sample_X, sample_y = [], []
        for _ in range(len(X)):
            index = int(random_float(0, len(X) - 1))
            sample_X.append(X[index])
            sample_y.append(y[index])
        return sample_X, sample_y
    
    def train_tree(X, y):
        return decision_tree_classifier(X, y, X)  # Training a tree on bootstrapped data
    
    trees = []
    for _ in range(num_trees):
        sample_X, sample_y = bootstrap_sample(train_X, train_y)
        tree = train_tree(sample_X, sample_y)
        trees.append(tree)

    predictions = []
    for row in test_X:
        tree_predictions = [tree[0] for tree in trees]  # Collect predictions from each tree
        predictions.append(max(set(tree_predictions), key=tree_predictions.count))
    
    return predictions

# Support Vector Machine (SVM) - Simplified Linear SVM
def svm(train_X, train_y, test_X, learning_rate=0.1, epochs=1000):
    def update_weights(w, b, x, y):
        if y * (sum(w_i * x_i for w_i, x_i in zip(w, x)) + b) < 1:
            w = [w_i + learning_rate * (y * x_i) for w_i, x_i in zip(w, x)]
            b = b + learning_rate * y
        return w, b

    # Initialize weights and bias
    w = [0] * len(train_X[0])
    b = 0
    for _ in range(epochs):
        for x, y in zip(train_X, train_y):
            w, b = update_weights(w, b, x, y)

    # Make predictions
    predictions = []
    for x in test_X:
        prediction = 1 if sum(w_i * x_i for w_i, x_i in zip(w, x)) + b >= 0 else -1
        predictions.append(prediction)
    
    return predictions

# Neural Network (Simple Perceptron with one hidden layer)
def neural_network(train_X, train_y, test_X, hidden_layer_size=5, epochs=1000, learning_rate=0.01):
    def sigmoid(x):
        return 1 / (1 + exp_neg(x))

    def sigmoid_derivative(x):
        return x * (1 - x)

    # Initialize weights and biases
    input_size = len(train_X[0])
    output_size = 1
    hidden_layer_weights = [[random_float(0, 1) for _ in range(input_size)] for _ in range(hidden_layer_size)]
    output_layer_weights = [random_float(0, 1) for _ in range(hidden_layer_size)]
    hidden_layer_biases = [random_float(0, 1) for _ in range(hidden_layer_size)]
    output_layer_bias = random_float(0, 1)

    # Train the model
    for epoch in range(epochs):
        for x, y in zip(train_X, train_y):
            # Forward pass
            hidden_layer_output = [sigmoid(sum(w * xi for w, xi in zip(weights, x)) + bias)
                                   for weights, bias in zip(hidden_layer_weights, hidden_layer_biases)]
            output_layer_input = sum(w * h for w, h in zip(output_layer_weights, hidden_layer_output)) + output_layer_bias
            output_layer_output = sigmoid(output_layer_input)

            # Backpropagation
            error = y - output_layer_output
            output_layer_delta = error * sigmoid_derivative(output_layer_output)
            hidden_layer_deltas = [output_layer_delta * w * sigmoid_derivative(h)
                                   for w, h in zip(output_layer_weights, hidden_layer_output)]

            # Update weights and biases
            output_layer_weights = [w + learning_rate * output_layer_delta * h for w, h in zip(output_layer_weights, hidden_layer_output)]
            output_layer_bias += learning_rate * output_layer_delta

            for i in range(hidden_layer_size):
                hidden_layer_weights[i] = [w + learning_rate * hidden_layer_deltas[i] * xi for w, xi in zip(hidden_layer_weights[i], x)]
                hidden_layer_biases[i] += learning_rate * hidden_layer_deltas[i]

    # Make predictions
    predictions = []
    for x in test_X:
        hidden_layer_output = [sigmoid(sum(w * xi for w, xi in zip(weights, x)) + bias)
                               for weights, bias in zip(hidden_layer_weights, hidden_layer_biases)]
        output_layer_input = sum(w * h for w, h in zip(output_layer_weights, hidden_layer_output)) + output_layer_bias
        output_layer_output = sigmoid(output_layer_input)
        predictions.append(1 if output_layer_output >= 0.5 else 0)
    
    return predictions

# Helper functions for the randomization and math from scratch

def random_float(low, high):
    return low + (high - low) * pseudo_random()

def pseudo_random():
    state = 123456789
    state = (state * 6364136223846793005 + 1) & 0xFFFFFFFFFFFFFFFF
    return (state >> 16) / 2**32

def exp_neg(x):
    return 1 / (1 + 2.718 ** (-x))


    # Test Decision Tree Classifier
    dt_predictions = decision_tree_classifier(train_X, train_y, test_X)
    dt_accuracy, dt_f1 = evaluate_predictions(dt_predictions, test_y)
    print(f"Decision Tree -> Accuracy: {dt_accuracy:.2f}%, F1-Score: {dt_f1:.2f}")
