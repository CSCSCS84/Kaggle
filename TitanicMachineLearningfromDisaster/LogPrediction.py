import numpy
import pandas
from statistics import mean
from statistics import variance
from TitanicMachineLearningfromDisaster.TitanicInstance import TitanicInstanceCreator
from TitanicMachineLearningfromDisaster import CrossValidation


class LogPrediction:
    def __init__(self, logValues=None):
        if logValues is None:
            self.logValues = {}
        else:
            self.logValues = logValues

    def fit(self, train, features):
        maxProb = 0.999
        for f in features:
            if (f != 'Survived' and f!='Sex'):
                data = train[train[f] == 1]
                count = data[(train['Survived'] == 1)].shape[0]

                prob = count / data.shape[0]

                if prob == 1.0:
                    prob = maxProb

                logProb = numpy.log(prob / (1 - prob))

                self.logValues.update({f: logProb})

    def predict(self, test, features):
        result = pandas.DataFrame(index=test.index)
        result['Survived'] = numpy.ones((test.shape[0],)) * (-1)
        for index, row in test.iterrows():
            prob = 0
            for f in features:
                if f != 'Survived':
                    if row[f] == 1:
                        prob += self.logValues.get(f)
            print(prob)
            if prob <= 0:
                result['Survived'][index] = 0
            else:
                result['Survived'][index] = 1
        return result

    def validate(self, train, sizeOfSamples, features):
        y = train['Survived']
        subsamples = self.calculateSubsamples(train, sizeOfSamples)
        scores = []
        for sample in subsamples:
            sample = sample.sort_index();
            trainData = train.drop(sample.index)
            yTrainData = y.drop(sample.index)
            self.fit(trainData, features)

            ySample = y.drop(trainData.index)
            score = self.score(sample, ySample, features)
            scores.append(score)
        return scores

    def calculateSubsamples(self, train, sizeOfSamples):
        subsamples = []
        permuted_indices = numpy.random.permutation(len(train))
        for i in range(sizeOfSamples):
            subsamples.append(train.iloc[permuted_indices[i::sizeOfSamples]])
        return subsamples

    def score(self, X, y, features):
        yPredict = self.predict(X, features)
        yP = yPredict.values
        yv = numpy.reshape(y.values, (y.values.shape[0], 1))
        diff = numpy.abs(numpy.subtract(yP, yv))
        if len(diff) == 0:
            score = 1.0
        else:
            score = (len(diff) - diff.sum()) / (len(diff))
        return score


fileNameExtension = 'ABCFGHIJKPQQRTUVY2'
fileNameExtensionTest = 'ABCFGHIJKPQQRTUVY'
featureNumber = 5
titanic = TitanicInstanceCreator.createInstance(fileNameExtension, fileNameExtensionTest, featureNumber)
l = LogPrediction()
l.fit(titanic.train, titanic.features)
# print(l.logValues)

sumScore = 0
n = 10

for i in range(0, n):

    scores = l.validate(titanic.train[titanic.features], 10, titanic.features)
    sumScore += mean(scores)
score=sumScore / n
print("%.4f" % (score))


result = l.predict(titanic.test, titanic.features)
result.to_csv('Data/Output/LogPrediction.csv', header='PassengerId\tSurvived', sep=',')
