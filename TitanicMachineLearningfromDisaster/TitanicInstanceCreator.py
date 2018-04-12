from TitanicMachineLearningfromDisaster.DataWorker import DataReader
from TitanicMachineLearningfromDisaster import TitanicInstance


def createInstance(fileNameExtensionTrain, fileNameExtensionTest, featureNumber):
    train = DataReader.getTrainData(fileNameExtensionTrain)
    test = DataReader.getTestData(fileNameExtensionTrain, fileNameExtensionTest)
    features = DataReader.getFeatures(fileNameExtensionTrain, featureNumber)
    titanic = TitanicInstance.TitanicInstance(train, test, features)
    return titanic
