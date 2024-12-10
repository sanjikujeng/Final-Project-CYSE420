# Decision Tree Implementation from Scratch
class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = {}

    def gini_impurity(self, groups, classes):
        n_instances = float(sum([len(group) for group in groups]))
        gini = 0.0
        for group in groups:
            size = len(group)
            if size == 0:
                continue
            score = 0.0
            for class_val in classes:
                proportion = [row[-1] for row in group].count(class_val) / size
                score += proportion ** 2
            gini += (1 - score) * (size / n_instances)
        return gini

    def split(self, index, value, dataset):
        left, right = [], []
        for row in dataset:
            if row[index] < value:
                left.append(row)
            else:
                right.append(row)
        return left, right

    def get_split(self, dataset):
        class_values = list(set(row[-1] for row in dataset))
        b_index, b_value, b_score, b_groups = 999, 999, 999, None
        for index in range(len(dataset[0]) - 1):
            for row in dataset:
                groups = self.split(index, row[index], dataset)
                gini = self.gini_impurity(groups, class_values)
                if gini < b_score:
                    b_index, b_value, b_score, b_groups = index, row[index], gini, groups
        return {'index': b_index, 'value': b_value, 'groups': b_groups}

    def build_tree(self, dataset, depth):
        left, right = dataset['groups']
        del(dataset['groups'])
        if not left or not right:
            dataset['left'] = dataset['right'] = self.terminal(left + right)
            return dataset
        if depth >= self.max_depth:
            dataset['left'], dataset['right'] = self.terminal(left), self.terminal(right)
            return dataset
        dataset['left'] = self.get_split(left)
        self.build_tree(dataset['left'], depth + 1)
        dataset['right'] = self.get_split(right)
        self.build_tree(dataset['right'], depth + 1)

    def terminal(self, group):
        outcomes = [row[-1] for row in group]
        return max(set(outcomes), key=outcomes.count)

    def fit(self, dataset):
        self.tree = self.get_split(dataset)
        self.build_tree(self.tree, 1)

    def predict(self, node, row):
        if row[node['index']] < node['value']:
            if isinstance(node['left'], dict):
                return self.predict(node['left'], row)
            else:
                return node['left']
        else:
            if isinstance(node['right'], dict):
                return self.predict(node['right'], row)
            else:
                return node['right']
