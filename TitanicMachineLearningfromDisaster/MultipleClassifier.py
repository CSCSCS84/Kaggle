import numpy
import pandas

#Predection using multiple classifier

class MultipleClassifier:

    yPredictions=None;

    def __init__(self, classifiers):
        self.classifiers=classifiers


    def fit(self, trainData, y):
        for classifier in self.classifiers:
            classifier.fit(trainData, y)

    def predict( self,testData):
        yResult = pandas.DataFrame(index=testData.index)
        ySum = numpy.zeros((1, testData.shape[0]));
        for classifier in self.classifiers:
            yPred = classifier.predict(testData)
            ySum = ySum + yPred
            yResult[classifier.__class__.__name__] = yPred;
        a = len(self.classifiers) / 2
        ytrans = ySum.transpose()
        self.yPredictions = yResult
        yPrediction = [0 if e <= len(self.classifiers) / 2 else 1 for e in ySum.transpose()]
        #yResult['FinalPrediction'] = yPrediction
        #yResult.to_csv('Data/Output/PredictedResultMultipleLeastCorrelated.csv', header=yResult.columns, sep=',')
        return yPrediction

    def score(self, X, y):
        yPredict = self.predict(X)
        diff = numpy.abs(yPredict - y.values)
        score = (len(diff) - diff.sum()) / (len(diff))
        return score

