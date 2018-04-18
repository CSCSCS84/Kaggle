from TitanicMachineLearningfromDisaster.ClassifierTuning import ClassifierTuner
from sklearn.tree import DecisionTreeClassifier
from TitanicMachineLearningfromDisaster import TitanicInstanceCreator
from sklearn.svm import SVC

fileNameExtension = 'ABCEFGHI12'
fileNameExtensionTest = 'ABCEFGHI1'
featureNumber = 1
titanic = TitanicInstanceCreator.createInstance(fileNameExtension, fileNameExtensionTest, featureNumber)
searchGrid = ClassifierTuner.getSVCGrid()
classifier = SVC(probability=True)
tunedClassifier = ClassifierTuner.tuneClassifier(titanic, classifier, searchGrid)
print(tunedClassifier)
ClassifierTuner.saveTunedClassifier(tunedClassifier, fileNameExtension, featureNumber)