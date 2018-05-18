from statistics import mean
from statistics import variance
import numpy
import pandas
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from TitanicMachineLearningfromDisaster.ClassifierTuning import ClassifierFactory
from TitanicMachineLearningfromDisaster import MultipleClassifier
from TitanicMachineLearningfromDisaster.TitanicInstance import TitanicInstanceCreator
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold



def votingClassifier(titanic, fileNameExtension, featureNumber):
    classifiers = getClassifiers(fileNameExtension, featureNumber)
    classifierName = [c.__class__.__name__ for c in classifiers]
    est = zip(classifierName, classifiers)
    votingC = VotingClassifier(estimators=list(est), voting='hard', n_jobs=4)

    votingC = votingC.fit(titanic.train, titanic.y)

    k_fold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)

    score = cross_val_score(votingC, titanic.train, titanic.y, cv=k_fold, n_jobs=1)

    print(score)
    print(mean(score))
    print(variance(score))
    test_Survived = pandas.Series(votingC.predict(titanic.test), name="Survived")
    yPredicted = votingC.predict(titanic.test)
    result = pandas.DataFrame(index=titanic.test.index)
    yPredicted = [int(i) for i in yPredicted]
    result['Survived'] = yPredicted
    return result




fileNameExtension = 'ABCFGHIJKPQQRTUVY2'
fileNameExtensionTest = 'ABCFGHIJKPQQRTUVY'
featureNumber = 8
titanic = TitanicInstanceCreator.createInstance(fileNameExtension, fileNameExtensionTest, featureNumber)

#print(titanic.test.isnull().sum())
#print(titanic.test)
#start(titanic, fileNameExtension, featureNumber=2)

result = votingClassifier(titanic, fileNameExtension, featureNumber)
result.to_csv('Data/Output/PredictedResultsVotingClassifier.csv', header='PassengerId\tSurvived', sep=',')

