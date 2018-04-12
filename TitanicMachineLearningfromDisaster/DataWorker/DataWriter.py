import json
from TitanicMachineLearningfromDisaster.DataWorker import FilenameBuilder


def writeTunedClassifierToJson(tunedClassifier, fileNameExtension, featureNumber):
    path = FilenameBuilder.getClassifierPath()
    classifierName = tunedClassifier.classifier.__class__.__name__
    filename = "%s/%s/%sTuned_%s_%s_%.4f.json" % (
        path, classifierName, classifierName, fileNameExtension, featureNumber, tunedClassifier.kfoldScore)
    f = open(filename, "w")

    classifierJson = json.dumps(tunedClassifier.classifier.get_params())
    f.write(classifierJson)
    f.close()
