
from TitanicMachineLearningfromDisaster.TitanicInstance import TitanicInstanceCreator

class LogPrediction:
    def __init__(self,logValues):
        if id is None:
            self.logValues={}
        else:
            self.logValues=logValues


    def fit(self,train,features):
        for f in features:
            data=train[f]
            count=train[(train[f] == 1)].shape[0]

            print(count)



fileNameExtension = 'ABCFGHIJKPQQRTUV2'
fileNameExtensionTest = 'ABCFGHIJKPQQRTUV'
featureNumber = 6
titanic = TitanicInstanceCreator.createInstance(fileNameExtension, fileNameExtensionTest, featureNumber)