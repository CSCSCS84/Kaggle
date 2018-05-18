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

def getIncorrectPredictionFilename(fileNameExtensionTrain,featureNumber):
    filename01 = "%s/Data/Output/Incorrect01%s_%s.csv" % (
        definitions.ROOT_DIR, fileNameExtensionTrain, featureNumber)

    filename10 = "%s/Data/Output/Incorrect10%s_%s.csv" % (
        definitions.ROOT_DIR, fileNameExtensionTrain, featureNumber)

    return [filename01,filename10]


def getClassifierPath():
    return "%s/Data/TunedClassifiers" % (definitions.ROOT_DIR)

def getOriginalDataPath():
    return "%s/Data/Input" % (definitions.ROOT_DIR)

def getOriginalTrainPath():
    return "%s/train.csv" % (getOriginalDataPath())

def getOriginalTestPath():
    return "%s/test.csv" % (getOriginalDataPath())

def getOriginalFeaturesPath():
    return "%s/features.txt" % (getOriginalDataPath())

def getRootPath():
    return "%s" % (definitions.ROOT_DIR)