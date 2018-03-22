import numpy
import pandas

#Predection using multiple classifier

def fit(classifiers, trainData, y):
    for classifier in classifiers:
        classifier.fit(trainData, y)

def predict(classifiers, testData):
    ySum = numpy.zeros((1,testData.shape[0]));
    for classifier in classifiers:
        yPred = classifier.predict(testData)
        ySum = ySum + yPred

    yPrediction = [0 if e <= len(classifiers)/2 else 1 for e in ySum.transpose()]
    return yPrediction
