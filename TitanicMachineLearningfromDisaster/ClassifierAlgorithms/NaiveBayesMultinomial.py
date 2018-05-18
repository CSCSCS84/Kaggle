import numpy
from TitanicMachineLearningfromDisaster.TitanicInstance import TitanicInstanceCreator
import pandas

class NaiveBayesMultinomial:
    def __init__(self, alpha=None):
        if alpha is None:
            self.alpha = 1.0
        else:
            self.alpha = alpha
        self.parameters = None

    def fit(self, X, y):
        self.featureCountZeroZero=pandas.DataFrame(columns=X.columns)

        self.featureCountOneZero = pandas.DataFrame(columns=X.columns)

        self.featureCountZeroOne = pandas.DataFrame(columns=X.columns)
        self.featureCountOneOne = pandas.DataFrame(columns=X.columns)


        for f in X.columns:

            yOne=y[y==1]
            yZero=y[y==0]

            yOneX=X.loc(yOne)






        return



fileNameExtension = 'ABCFGHIJKPRTVYQQU'
fileNameExtensionTest = 'ABCFGHIJKPRTVYQQU'
featureNumber = 8
titanic = TitanicInstanceCreator.createInstance(fileNameExtension, fileNameExtensionTest, featureNumber)

bayes=NaiveBayesMultinomial()
bayes.fit(titanic.train[titanic.features],titanic.y)