from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from TitanicMachineLearningfromDisaster.DataWorker import DataWriter
from TitanicMachineLearningfromDisaster.ClassifierTuning import TunedClassifier


def tuneClassifier(titanic, classifier, grid):
    kfold = StratifiedKFold(n_splits=10)
    gridSearch = GridSearchCV(classifier, param_grid=grid, cv=kfold, scoring="accuracy", n_jobs=4, verbose=1)
    gridSearch.fit(titanic.train, titanic.y)
    return TunedClassifier.TunedClassifier(gridSearch.best_estimator_, gridSearch.best_score_)


def saveTunedClassifier(tunedClassifier, fileNameExtension, featureNumber):
    DataWriter.writeTunedClassifierToJson(tunedClassifier, fileNameExtension, featureNumber)


def getRandomForestGrid():
    return {"random_state": [9],
            "max_depth": [14],
            "max_features": [2, 3, 4],
            "min_samples_split": [2, ],
            "min_samples_leaf": [2],
            "bootstrap": [False],
            "n_estimators": [66],
            "criterion": ["entropy"]}


def getDecisionTreeGrid():
    return {"max_depth": [9],
            "max_features": [4],
            "min_samples_split": [4],
            "min_samples_leaf": [1, 2],
            "min_weight_fraction_leaf": [0.0, 0.005],
            "min_impurity_decrease": [0.0, 0.001],
            "random_state": [2, 3, 4],
            "presort": [False, True],
            "criterion": ['entropy']
            }

def getSVCGrid():
    return {'kernel': ['rbf'],
                      'gamma': [0.006],
                      'C': [2],
                      'degree':[1,2],
                      'random_state':[True],
                      'coef0':[0.0],
                      'cache_size':[85,90],
                      'class_weight':['balanced']
            }