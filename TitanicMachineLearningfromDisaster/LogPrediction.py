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
        maxProb = 0.9
        minProb=0.1
        for f in features:
            if (f != 'Survived' and f != 'Sex'):
                data = train[train[f] == 1]
                count = data[(train['Survived'] == 1)].shape[0]

                prob = count / data.shape[0]

                if prob > maxProb:
                    prob = maxProb
                if prob < minProb:
                    prob=minProb

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
            # if abs(prob)<0.2:
            #    print(row)


            #print("%s" % (prob))
            if prob <= 0.0:
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

    def incorrectPrediction(self, train, sizeOfSamples, features):
        y = train['Survived']
        subsamples = self.calculateSubsamples(train, sizeOfSamples)
        incorrectPred01 = []
        incorrectPred10 = []
        for sample in subsamples:
            sample = sample.sort_index();
            trainData = train.drop(sample.index)
            yTrainData = y.drop(sample.index)
            self.fit(trainData, features)

            ySample = y.drop(trainData.index)
            inCor = self.findIncorrect(sample, ySample, features)
            incorrectPred01.extend(inCor[0])
            incorrectPred10.extend(inCor[1])
        incorrect01=pandas.DataFrame()
        incorrect01['PassengerId']=incorrectPred01

        incorrect10=pandas.DataFrame()
        incorrect10['PassengerId']=incorrectPred10
        return [incorrect01,incorrect10]

    def findIncorrect(self, X, y, features):
        yPredict = self.predict(X, features)
        incor01 = []
        incor10=[]
        for index, row in yPredict.iterrows():
            if row['Survived'] != y.ix[index]:
                if row['Survived']==0:
                    incor01.append(index)
                else:
                    incor10.append(index)
        return [incor01,incor10]

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
scoreVal = sumScore / n
print("%.4f" % (scoreVal))

incorrectPred = l.incorrectPrediction(titanic.train[titanic.features], 10, titanic.features)
#print(incorrectPred)
incorrectPred[0].to_csv('Data/Output/Incorrect01%s_%s.csv' % (fileNameExtension,featureNumber))
incorrectPred[1].to_csv('Data/Output/Incorrect10%s_%s.csv' % (fileNameExtension,featureNumber))
print()
result = l.predict(titanic.test, titanic.features)
result.to_csv('Data/Output/LogPrediction.csv', header='PassengerId\tSurvived', sep=',')
