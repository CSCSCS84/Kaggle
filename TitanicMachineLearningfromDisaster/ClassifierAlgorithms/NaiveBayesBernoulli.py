from TitanicMachineLearningfromDisaster.TitanicInstance import TitanicInstanceCreator
import pandas
import numpy

# Implementation of Naive Bayes with Bernoulli distributed datas.
class NaiveBayesBernoulli:

    def fit(self, X, y):
        self.pxiOne = self.probalitiesForFeatures(X, y, 1)
        self.pxiZero = self.probalitiesForFeatures(X, y, 0)
        self.yProb = y.sum() / X.shape[0]

    def probalitiesForFeatures(self, X, y, yi):
        yiIndexes = y[y == yi].index
        XSubset = X.ix[yiIndexes]
        countValues = XSubset.apply(pandas.value_counts)
        countValues = countValues.fillna(0.0)

        numberOfData = yiIndexes.shape[0]
        pxi = pandas.Series(index=X.columns)
        for feature in X.columns:
            pxi[feature] = countValues[feature][1.0] / numberOfData
        return pxi

    def predict(self, test):
        yPredicted = numpy.ndarray(test.shape[0])
        i = 0
        for index, row in test.iterrows():
            yZero = self.calcLikelihood(test, 0, row)
            yOne = self.calcLikelihood(test, 1, row)
            yPredicted[i] = 0 if (yZero >= yOne) else 1
            i += 1
        return yPredicted

    def calcLikelihood(self, X, y, row):
        likelihood = y * self.yProb + (1 - y) * (1 - self.yProb)
        for f in X.columns:
            pxi = self.calculatePxi(f, row, y)
            likelihood *= pxi
        return likelihood

    def calculatePxi(self, feature, row, y):
        if y == 0:
            piy = self.pxiZero[feature]
        else:
            piy = self.pxiOne[feature]
        return piy * row[feature] + (1 - piy) * (1 - row[feature])


fileNameExtension = 'ABCFGHIJKPRTVYQQU'
fileNameExtensionTest = 'ABCFGHIJKPRTVYQQU'
featureNumber = 9
titanic = TitanicInstanceCreator.createInstance(fileNameExtension, fileNameExtensionTest, featureNumber)

bayes = NaiveBayesBernoulli()
bayes.fit(titanic.train[titanic.features], titanic.y)
yPredicted = bayes.predict(titanic.test)

result = pandas.DataFrame(index=titanic.test.index)
result['Survived'] = yPredicted
result.to_csv('../Data/Output/NaiveBayesClassifier.csv', header='PassengerId\tSurvived', sep=',')
