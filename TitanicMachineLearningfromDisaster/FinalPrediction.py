from statistics import mean
import numpy
import pandas
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from TitanicMachineLearningfromDisaster.ClassifierTuning import ClassifierFactory
from TitanicMachineLearningfromDisaster import MultipleClassifier
from TitanicMachineLearningfromDisaster.TitanicInstance import TitanicInstanceCreator
from sklearn.model_selection import KFold, cross_val_score
from statistics import mean
from sklearn.model_selection import StratifiedKFold


def start(titanic, fileNameExtension, featureNumber):
    result = firstLevelPrediction(titanic, fileNameExtension, featureNumber)
    # return
    result.to_csv('Data/Output/PredictedResultsFirstLevel%s.csv' % (fileNameExtension), header='PassengerId\tSurvived', sep=',')
    # secondLevePrediction(train, test,y)


def firstLevelPrediction(titanic, fileNameExtension, featureNumber):
    classifiers = getClassifiers(fileNameExtension, featureNumber)
    fitAllClassifiers(classifiers, titanic.train, titanic.y)
    score = calcCrossValidationScore(classifiers, titanic.train, titanic.y)
    print(score)
    result = predictAll(classifiers, titanic.train)
    return result


def fitAllClassifiers(classifiers, trainData, y):
    for c in classifiers:
        c.fit(trainData, y.values.ravel())


def predictAll(classifiers, testData):
    result = pandas.DataFrame(index=testData.index)
    for c in classifiers:
        yAll = c.predict(testData)
        result[c.__class__.__name__] = yAll
    return result


def calcCrossValidationScore(classifiers, train, y):
    k_fold = StratifiedKFold(n_splits=10)
    scores = []
    for c in classifiers:
        s = cross_val_score(c, train, y, cv=k_fold, n_jobs=-1)
        scores.append(mean(s))

        print(c.__class__.__name__)
        #print(c)
        print(mean(s))
    return scores


def votingClassifier(titanic, fileNameExtension, featureNumber):
    classifiers = getClassifiers(fileNameExtension, featureNumber)
    classifierName = [c.__class__.__name__ for c in classifiers]
    est = zip(classifierName, classifiers)
    votingC = VotingClassifier(estimators=list(est), voting='soft', n_jobs=4)

    votingC = votingC.fit(titanic.train, titanic.y)
    k_fold = StratifiedKFold(n_splits=10)
    score = cross_val_score(votingC, titanic.train, titanic.y, cv=k_fold, n_jobs=1)
    print(mean(score))

    test_Survived = pandas.Series(votingC.predict(titanic.test), name="Survived")
    yPredicted = votingC.predict(titanic.test)
    result = pandas.DataFrame(index=titanic.test.index)
    yPredicted = [int(i) for i in yPredicted]
    result['Survived'] = yPredicted
    return result

def getClassifiers(fileNameExtension, featureNumber):
    classifiers = []

    classifiers.append(ClassifierFactory.constructClassifier("SVC", fileNameExtension, featureNumber, "0.9787"))
    return classifiers


def getClassifiers00(fileNameExtension, featureNumber):
    classifiers = []
    classifiers.append(ClassifierFactory.constructClassifier("SVC", fileNameExtension, featureNumber, "0.8309"))
    classifiers.append(
        ClassifierFactory.constructClassifier("AdaBoostClassifier", fileNameExtension, featureNumber, "0.8241"))
    classifiers.append(
        ClassifierFactory.constructClassifier("KNeighborsClassifier", fileNameExtension, featureNumber, "0.8253"))
    classifiers.append(
        ClassifierFactory.constructClassifier("GradientBoostingClassifier", fileNameExtension, featureNumber, "0.8512"))
    classifiers.append(
        ClassifierFactory.constructClassifier("LogisticRegression", fileNameExtension, featureNumber, "0.8309"))

    classifiers.append(
        ClassifierFactory.constructClassifier("MLPClassifier", fileNameExtension, featureNumber, "0.8275"))

    classifiers.append(
        ClassifierFactory.constructClassifier("LinearDiscriminantAnalysis", fileNameExtension, featureNumber, "0.8365"))
    classifiers.append(
        ClassifierFactory.constructClassifier("RandomForestClassifier", fileNameExtension, featureNumber, "0.8388"))
    classifiers.append(
        ClassifierFactory.constructClassifier("ExtraTreesClassifier", fileNameExtension, featureNumber, "0.8388"))

    classifiers.append(
        ClassifierFactory.constructClassifier("DecisionTreeClassifier", fileNameExtension, featureNumber, "0.8320"))
    return classifiers

def getClassifiers2(fileNameExtension, featureNumber):
    classifiers = []
    classifiers.append(ClassifierFactory.constructClassifier("SVC", fileNameExtension, featureNumber, "0.8309"))
    classifiers.append(
        ClassifierFactory.constructClassifier("ExtraTreesClassifier", fileNameExtension, featureNumber, "0.8388"))
    classifiers.append(
        ClassifierFactory.constructClassifier("GradientBoostingClassifier", fileNameExtension, featureNumber, "0.8512"))
    classifiers.append(
        ClassifierFactory.constructClassifier("MLPClassifier", fileNameExtension, featureNumber, "0.8275"))
    classifiers.append(
        ClassifierFactory.constructClassifier("LinearDiscriminantAnalysis", fileNameExtension, featureNumber, "0.8365"))
    classifiers.append(
        ClassifierFactory.constructClassifier("LogisticRegression", fileNameExtension, featureNumber, "0.8309"))
    classifiers.append(
        ClassifierFactory.constructClassifier("RandomForestClassifier", fileNameExtension, featureNumber, "0.8388"))
    classifiers.append(
        ClassifierFactory.constructClassifier("AdaBoostClassifier", fileNameExtension, featureNumber, "0.8241"))
    classifiers.append(
        ClassifierFactory.constructClassifier("DecisionTreeClassifier", fileNameExtension, featureNumber, "0.8320"))
    classifiers.append(
        ClassifierFactory.constructClassifier("KNeighborsClassifier", fileNameExtension, featureNumber, "0.8253"))

    return classifiers


fileNameExtension = 'ABCEFGHIJKL12'
fileNameExtensionTest = 'ABCEFGHIJKL1'
featureNumber = 1
titanic = TitanicInstanceCreator.createInstance(fileNameExtension, fileNameExtensionTest, featureNumber)


#start(titanic, fileNameExtension, featureNumber=1)
result = votingClassifier(titanic, fileNameExtension, featureNumber)
result.to_csv('Data/Output/PredictedResultsVotingClassifier.csv', header='PassengerId\tSurvived', sep=',')
