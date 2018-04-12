from TitanicMachineLearningfromDisaster import definitions


def getTrainFilename(fileNameExtensionTrain):
    filename = "%s/Data/Input/PreparedData/%s/PreparedTrain_%s.csv" % (
    definitions.ROOT_DIR, fileNameExtensionTrain, fileNameExtensionTrain)
    return filename


def getTestFilename(fileNameExtensionTrain, fileNameExtensionTest):
    filename = "%s/Data/Input/PreparedData/%s/PreparedTest_%s.csv" % (
    definitions.ROOT_DIR, fileNameExtensionTrain, fileNameExtensionTest)
    return filename


def getFeatureFilename(fileNameExtensionTrain, featureNumber):
    filename = "%s/Data/Input/PreparedData/%s/Features_%s.txt" % (
    definitions.ROOT_DIR, fileNameExtensionTrain, featureNumber)
    return filename


def getClassifierPath():
    return "%s/TunedClassifiers" % (definitions.ROOT_DIR)
