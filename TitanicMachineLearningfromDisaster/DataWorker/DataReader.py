from TitanicMachineLearningfromDisaster import DataWorker
import pandas
from TitanicMachineLearningfromDisaster.DataWorker import FilenameBuilder


def getTrainData(fileNameExtensionTrain):
    filename = FilenameBuilder.getTrainFilename(fileNameExtensionTrain)
    train = pandas.read_csv(filename,
                            index_col='PassengerId')
    return train


def getTestData(fileNameExtensionTrain, fileNameExtensionTest):
    filename = DataWorker.FilenameBuilder.getTestFilename(fileNameExtensionTrain, fileNameExtensionTest)
    test = pandas.read_csv(filename,
                           index_col='PassengerId')
    return test


def getFeatures(fileNameExtensionTrain, featureNumber):
    filename = DataWorker.FilenameBuilder.getFeatureFilename(fileNameExtensionTrain, featureNumber)
    text_file = open("%s" % (filename), "r")
    features = text_file.read().replace('\n', '').split(',')
    return features
