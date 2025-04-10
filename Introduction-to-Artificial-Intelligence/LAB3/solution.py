import sys
import math

data = []
set_values = {}
test = []
indices = list()
glo_features = []


class Node:
    def __init__(self, feature):
        self.feature = feature
        self.children = {}

    def add_child(self, value, subtree):
        self.children[value] = subtree

    def __repr__(self, level=1, prefix=""):
        ret = ""
        for value, subtree in self.children.items():
            new_prefix = f"{prefix}{level}:{self.feature}={value} "
            ret += subtree.__repr__(level + 1, new_prefix).strip() + "\n"
        return ret


class Leaf:
    def __init__(self, classification):
        self.classification = classification

    def __repr__(self, level=1, prefix=""):
        return f"{prefix}{self.classification}"


class ID3:
    def __init__(self, hyperparameters=None):
        self.hyperparameters = hyperparameters
        self.tree = None
        self.max_depth = None
        if hyperparameters and "max_depth" in hyperparameters:
            self.max_depth = hyperparameters["max_depth"]

    def fit(self, dataset, parent_dataset, features, depth=0):
        self.tree = self.id3(dataset, parent_dataset, features, depth)
        print("[BRANCHES]:")
        print(self.tree)

    def id3(self, dataset, parent_dataset, features, depth):
        if len(dataset) == 0:
            mf_label = self.most_frequent_label(parent_dataset)
            return Leaf(mf_label)
        mf_label = self.most_frequent_label(dataset)

        temp_data = []
        for data in dataset:
            if mf_label == data[-1]:
                temp_data.append(data)
        if temp_data == dataset or (self.max_depth is not None and depth >= self.max_depth):
            return Leaf(mf_label)

        if len(features) == 0:
            return Leaf(mf_label)

        chosen = self.most_discriminative_feature(dataset)
        tree = Node(chosen)
        values = set_values[chosen]
        feature_index = glo_features.index(chosen)
        for value in values:
            subset = [data for data in dataset if data[feature_index] == value]
            subtree = self.id3(subset, dataset, [f for f in features if f != chosen], depth + 1)
            tree.add_child(value, subtree)
        indices.remove(glo_features.index(chosen))
        return tree

    def predict(self, testset):
        results = []
        tests = []
        for test in testset:
            tests.append(test[-1])
        mf_label = self.most_frequent_label(testset)
        for test in testset:
            node = self.tree
            while isinstance(node, Node):
                value = test[glo_features.index(node.feature)]
                node = node.children.get(value)
                if node is None:
                    results.append(mf_label)
                    break
            results.append(node.classification if node else None)
        print(end="[PREDICTIONS]: ")
        for result in results:
            if result is None:
                results.remove(result)
        correct = 0
        index = 0
        for result in results:
            if result == testset[index][-1]:
                correct += 1
            print(end=result + " ")
            index += 1

        accuracy = correct / len(testset)
        print("\n[ACCURACY]: ", f"{accuracy:.5f}")

        values = sorted(set_values[glo_features[-1]])
        conf_matrix = [[]]
        for each1 in values:
            temp = []
            for each2 in values:
                count = 0
                for i in range(0, len(results)):
                    res = results[i]
                    test = tests[i]
                    if res + test == each2 + each1:
                        count += 1
                temp.append(count)
            conf_matrix.append(temp)
        conf_matrix.pop(0)
        print("[CONFUSION_MATRIX]: ")
        for each1 in range(0, len(values)):
            for each2 in range(0, len(values)):
                print(conf_matrix[each1][each2], end=" ")
            print()

    def most_discriminative_feature(self, dataset):
        global indices
        global glo_features

        E = self.entropy(dataset)
        number_of_lines = len(dataset)
        feature_IG = {}

        for i in range(len(glo_features) - 1):
            if i in indices:
                continue

            counter = {}
            for data in dataset:
                value = data[i]
                if value not in counter:
                    counter[value] = 0
                counter[value] += 1

            feature_entropy = 0
            for value in counter:
                subset = [data for data in dataset if data[i] == value]
                subset_entropy = self.entropy(subset)
                weight = counter[value] / number_of_lines
                feature_entropy += weight * subset_entropy

            IG = E - feature_entropy
            feature_IG[glo_features[i]] = IG
        print(feature_IG)
        chosen = max(feature_IG, key=feature_IG.get)
        indices.append(glo_features.index(chosen))
        return chosen

    def most_frequent_label(self, dataset):
        count_labes = {}
        for data in dataset:
            if data[-1] not in count_labes:
                count_labes[data[-1]] = 1
            else:
                count_labes[data[-1]] += 1
        sorted_count = {key: count_labes[key] for key in sorted(count_labes.keys())}
        most = 0
        mf_label = ""
        for label in sorted_count:
            if sorted_count[label] > most:
                most = sorted_count[label]
                mf_label = label

        return mf_label

    def take_subset(self, dataset, value, feature_index):
        return [data for data in dataset if data[feature_index] == value]

    def entropy(self, dataset):
        count = {}
        data_count = len(dataset)

        # Count occurrences of each label in the dataset
        for data in dataset:
            label = data[-1]
            if label not in count:
                count[label] = 0
            count[label] += 1

        entropy = 0
        for label_count in count.values():
            probability = label_count / data_count
            if probability != 0:
                entropy -= probability * math.log(probability, 2)

        return entropy


def main():
    global glo_features
    file1 = open(sys.argv[1])
    file2 = open(sys.argv[2])
    lines = file1.readlines()
    global data
    for line in lines:
        line = line.strip()
        data.append(line.split(","))
    lines = file2.readlines()
    for line in lines:
        line = line.strip()
        test.append(line.split(","))
    glo_features = data[0]
    data.pop(0)
    test.pop(0)
    index = 0
    for feature in glo_features:
        tempset = set()
        for each in data:
            tempset.add(each[index])
        set_values[feature] = tempset
        index += 1

    hyperparameters = None
    if len(sys.argv) > 3:
        max_depth = int(sys.argv[3])
        hyperparameters = {"max_depth": max_depth}

    model = ID3(hyperparameters)
    model.fit(data, data, glo_features)
    model.predict(test)


if __name__ == "__main__":
    main()
