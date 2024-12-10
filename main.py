data = [
    [5, 3, 0], [2, 8, 1], [7, 1, 0], [1, 6, 1], [4, 2, 0],
    [6, 5, 1], [3, 7, 1], [8, 2, 0], [5, 7, 1], [2, 3, 0]
]

train_data = data[:8]
test_data = data[8:]

train_X = [row[:-1] for row in train_data]
train_y = [row[-1] for row in train_data]
test_X = [row[:-1] for row in test_data]
test_y = [row[-1] for row in test_data]

def simple_classifier(features):
    feature_1, feature_2 = features
    return 1 if feature_1 + feature_2 > 8 else 0

predictions = [simple_classifier(features) for features in test_X]

correct = sum([1 for pred, actual in zip(predictions, test_y) if pred == actual])
accuracy = correct / len(test_y) * 100

print("Test Data:", test_data)
print("Predictions:", predictions)
print(f"Accuracy: {accuracy:.2f}%")
