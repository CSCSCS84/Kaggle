import numpy
import pandas


def analyse(features):
    wrongPrediction = pandas.read_csv("WrongPrediction.csv", index_col='PassengerId')
    train = pandas.read_csv("PreparedTrain.csv", index_col='PassengerId')
    wrongDecribe = wrongPrediction.describe()
    trainDescribe = train.describe()

    for f in features:
        print(wrongDecribe[f])
        print(trainDescribe[f])
        print("")
