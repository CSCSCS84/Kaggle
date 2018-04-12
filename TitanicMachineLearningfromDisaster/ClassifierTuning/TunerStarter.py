from TitanicMachineLearningfromDisaster.ClassifierTuning import ClassifierTuner
from sklearn.tree import DecisionTreeClassifier
from TitanicMachineLearningfromDisaster import TitanicInstanceCreator

fileNameExtension = 'ABCEFGHI12'
fileNameExtensionTest = 'ABCEFGHI1'
featureNumber = 1
titanic = TitanicInstanceCreator.createInstance(fileNameExtension, fileNameExtensionTest, featureNumber)
searchGrid = ClassifierTuner.getDecisionTreeGrid()
classifier = DecisionTreeClassifier()
tunedClassifier = ClassifierTuner.tuneClassifier(titanic, classifier, searchGrid)
print(tunedClassifier)
ClassifierTuner.saveTunedClassifier(tunedClassifier, fileNameExtension, featureNumber)