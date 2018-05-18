from TitanicMachineLearningfromDisaster.DataWorker import FilenameBuilder
import json
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB


def constructClassifier(classifierName, fileNameExtension, featureNumber, score):
    classifier = getClassifier(classifierName)
    if classifier == None:
        print("No classifier found with name %s found" % (classifierName))
        return None
    path = FilenameBuilder.getClassifierPath()

    filename = "%s/%s/%sTuned_%s_%s_%s.json" % (
        path, classifierName, classifierName, fileNameExtension, featureNumber, score)
    data = json.load(open(filename))

    classifier.set_params(**data)

    return classifier


def getClassifier(classifierName):
    if classifierName == "RandomForestClassifier":
        return RandomForestClassifier()
    elif classifierName == "DecisionTreeClassifier":
        return DecisionTreeClassifier()
    elif classifierName == "SVC":
        return SVC()
    elif classifierName == "ExtraTreesClassifier":
        return ExtraTreesClassifier()
    elif classifierName == "MLPClassifier":
        return MLPClassifier()
    elif classifierName == "GradientBoostingClassifier":
        return GradientBoostingClassifier()
    elif classifierName == "LinearDiscriminantAnalysis":
        return LinearDiscriminantAnalysis()
    elif classifierName == "LogisticRegression":
        return LogisticRegression()
    elif classifierName == "KNeighborsClassifier":
        return KNeighborsClassifier()
    elif classifierName == "AdaBoostClassifier":
        d = DecisionTreeClassifier()
        return AdaBoostClassifier(d)
    elif classifierName == "GaussianNB":
        return GaussianNB()
    elif classifierName == "MultinomialNB":
        return MultinomialNB()
    elif classifierName == "BernoulliNB":
        return BernoulliNB()
    else:
        return None
