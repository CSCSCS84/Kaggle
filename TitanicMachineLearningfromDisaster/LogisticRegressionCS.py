import numpy
import pandas
import collections
#Logistic Regression using Gradient Descent Method. Maximal Iteration and error tolerance can be chosen.

class LogisticRegressionCS:
    def __init__(self, max_iter=None, tolerance=None, alpha=None):
        if max_iter is None:
            self.max_iter = 100
        else:
            self.max_iter = max_iter
        if tolerance is None:
            self.tolerance = 0.00001
        else:
            self.tolerance = tolerance

        self.weight = None;

        if alpha is None:
            self.alpha = 0.001
        else:
            self.alpha = alpha

    def fit(self, traindata, y):
        self.weight = self.gradientDescent(traindata, y)

    def predict(self, data):
        probability = self.calculateEstimatedProbability(self.weight, data)
        y = self.calculatePrediction(probability)
        return y

    def calculatePrediction(self, probability):
        y = numpy.where(probability < 0.5, 0, 1)
        return y

    def gradientDescent(self, X, y):
        numberOfFeatures = X.shape[1]
        weight = pandas.DataFrame(numpy.ones((numberOfFeatures, 1)), index=X.columns)
        Xtranpose = X.transpose()
        cost = float("inf");

        for i in range(1, self.max_iter):
            if cost > self.tolerance:
                weight = self.updateWeight(X, Xtranpose, weight, numberOfFeatures, y)
                cost = self.calculateCosts(X, weight, y)
            else:
                break

        return weight

    def updateWeight(self, X, Xtranpose, weight, numberOfFeatures, y):
        A = Xtranpose.multiply(self.alpha / numberOfFeatures)
        C = X.dot(weight)
        B = self.sigmoid(C).values - y
        weight = weight.values - A.dot(B)
        return weight

    def calculateCosts(self, X, weight, y):
        cost = 0
        A = self.sigmoid(X.dot(weight))
        for i in range(1, X.shape[1]):
            cost += y.values[i] * numpy.log(A.values[i])
            +(1 - y.values[i]) * numpy.log((1 - A.values[i]))

        cost *= -1 / len(y)
        return cost;

    # use weight to prognose
    def calculateEstimatedProbability(self, weight, X):
        prob = self.sigmoid(X.dot(weight))
        return prob

    def sigmoid(self, x):
        return 1.0 / (1 + numpy.exp(-x))

    def score(self, X, y):
        yPredict = self.predict(X)
        diff = numpy.abs(yPredict - y.values)
        score = (len(diff) - diff.sum()) / (len(diff))
        return score

