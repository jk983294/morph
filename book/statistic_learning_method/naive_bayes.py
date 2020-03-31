import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import math


def create_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    data = np.array(df.iloc[:100, :])
    return data[:, :-1], data[:, -1]


class NaiveBayes:
    def __init__(self):
        """self.model[label] = [(mean, stddev) of each feature]"""
        self.model = None

    @staticmethod
    def mean(X):
        return sum(X) / float(len(X))

    @staticmethod
    def stddev(X):
        avg = NaiveBayes.mean(X)
        return math.sqrt(sum([pow(x - avg, 2) for x in X]) / float(len(X)))

    @staticmethod
    def gaussian_probability(x, mean, stddev):
        """gaussian density value, calculate each individual feature probability under gaussian assumption"""
        exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stddev, 2))))
        return (1 / (math.sqrt(2 * math.pi) * stddev)) * exponent

    @staticmethod
    def summarize(train_data):
        """get each feature's mean stddev"""
        summaries = [(NaiveBayes.mean(i), NaiveBayes.stddev(i)) for i in zip(*train_data)]
        return summaries

    def fit(self, X, y):
        labels = list(set(y))
        data = {label: [] for label in labels}
        for f, label in zip(X, y):
            data[label].append(f)
        self.model = {
            label: self.summarize(sample) for label, sample in data.items()
        }
        return 'gaussianNB train done!'

    def calculate_probabilities(self, input_data):
        probabilities = {}      # label -> probability
        for label, value in self.model.items():
            probabilities[label] = 1
            for i in range(len(value)):
                mean, stddev = value[i]
                """NB assumption is each feature is independent so we can multiply each feature's probability"""
                probabilities[label] *= self.gaussian_probability(input_data[i], mean, stddev)
        return probabilities

    def predict(self, X_test):
        """sort by probability, choose label with highest probability"""
        label = sorted(self.calculate_probabilities(X_test).items(), key=lambda x: x[-1])[-1][0]
        return label

    def score(self, X_test, y_test):
        right = 0
        for X, y in zip(X_test, y_test):
            label = self.predict(X)
            if label == y:
                right += 1
        return right / float(len(X_test))


X, y = create_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

model = NaiveBayes()
model.fit(X_train, y_train)
print("predicted label:", model.predict([4.4,  3.2,  1.3,  0.2]))
print("predict precision on test dataset:", model.score(X_test, y_test))
