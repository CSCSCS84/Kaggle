import numpy
import pandas
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from TitanicMachineLearningfromDisaster import MultipleClassifier
from TitanicMachineLearningfromDisaster import CrossValidation
from TitanicMachineLearningfromDisaster import LogisticRegressionCS
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import seaborn as sns
import matplotlib.pyplot as plt
from statistics import mean
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

def start(train,test, y):

    #y = pandas.DataFrame(train['Survived'])
    trainData = train
    testData = test
    mc=MultipleClassifier.MultipleClassifier(getClassifiersLeastCorrelated())
    classifiers=[mc]

    fitClassifiers(classifiers, trainData, y)
    scores=crossValidate(classifiers, trainData, y, 10,10)
    for s in scores:
        print(s)
    result=predict(classifiers, trainData)


    result.to_csv('Data/Output/PredictedResults%s.csv' % (fileNameExtension), header='PassengerId\tSurvived', sep=',')

def calcCorrelation(trainData,features):
    correlation = trainData[features].corr().round(
        2)
    print(correlation)
    ax = sns.heatmap(correlation, annot=True)
    plt.show()
    correlation.to_csv('Correlation2.csv', header=features, sep=',')

def fitClassifiers(classifiers, trainData, y):
    for c in classifiers:
        c.fit(trainData, y.values.ravel())

def crossValidate(classifiers, trainData, y, numOfValidations,k):
    scores = []

    for c in classifiers:
        meanSum=0
        for i in range(0, numOfValidations):
            score = CrossValidation.validate(trainData, k, c, y)
            print(score)
            meanSum+= mean(score)
        average=numpy.round(meanSum/numOfValidations,3)
        scores.append(average)

    return scores

def predict(classifiers, testData):
    result = pandas.DataFrame(index=testData.index)
    for c in classifiers:
        yAll = c.predict(testData)
        result[c.__class__.__name__] = yAll
    return result


def getClassifiers():
    classifiers = []
    classifiers.append(getSVCTuned())
    classifiers.append(getExtraTreeTuned())
    classifiers.append(getGradientBoostingTuned())
    classifiers.append(getMLPTuned())
    classifiers.append(getLinearDiscriminantTuned())
    classifiers.append(getLogisticRegressionTuned())
    classifiers.append(getRandomForestTuned())
    classifiers.append(getAdaboostTuned())
    classifiers.append(getDecisionTreeTuned())
    classifiers.append(getKNeighborsTuned())
    return classifiers

def getClassifiersLeastCorrelated():
    classifiers = []
    classifiers.append(getLinearDiscriminantTuned())
    classifiers.append(getLogisticRegressionTuned())
    classifiers.append(getAdaboostTuned())
    classifiers.append(getGradientBoostingTuned())
    classifiers.append(getKNeighborsTuned())
    return classifiers

def getRandomForestTuned():
    return RandomForestClassifier(bootstrap=False, class_weight=None, criterion='gini',
            max_depth=None, max_features=3, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=3, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=300, n_jobs=1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)

def randomForestTuning(train, y):
    kfold = StratifiedKFold(n_splits=10)
    RFC = RandomForestClassifier()

    ## Search grid for optimal parameters
    rf_param_grid = {"max_depth": [None],
                     "max_features": [4],
                     "min_samples_split": [ 3],
                     "min_samples_leaf": [3],
                     "bootstrap": [False],
                     "n_estimators": [80,90,100,110,120],
                     "criterion": ["gini"]}

    gsRFC = GridSearchCV(RFC, param_grid=rf_param_grid, cv=kfold, scoring="accuracy", n_jobs=4, verbose=1)

    gsRFC.fit(train, y)

    RFC_best = gsRFC.best_estimator_
    print(RFC_best)
    # Best score
    print(gsRFC.best_score_)


def getAdaboostTuned():
    return AdaBoostClassifier(algorithm='SAMME.R',
          base_estimator=DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=None,
            max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, presort=False, random_state=None,
            splitter='best'),
          learning_rate=0.15, n_estimators=4, random_state=7)


def calcCorrelationOfClassifiers():
    result = pandas.read_csv("Data/Output/PredictedResultsABCFGHI2.csv",
                             index_col='PassengerId')
    classifiers = ['SVC', 'DecisionTreeClassifier', 'AdaBoostClassifier', 'KNeighborsClassifier', 'MLPClassifier']
    correlation = result[classifiers].corr().round(
        2)
    print(correlation)
    ax = sns.heatmap(correlation, annot=True)
    plt.show()

def getExtraTreeTuned():
    return ExtraTreesClassifier(bootstrap=False, class_weight=None, criterion='gini',
           max_depth=None, max_features=5, max_leaf_nodes=None,
           min_impurity_decrease=0.0, min_impurity_split=None,
           min_samples_leaf=3, min_samples_split=4,
           min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=1,
           oob_score=False, random_state=None, verbose=0, warm_start=False)


def extraTreeTuning(train, y):
    ExtC=ExtraTreesClassifier()
    ## Search grid for optimal parameters
    kfold = StratifiedKFold(n_splits=10)
    ex_param_grid = {"max_depth": [None],
                     "max_features": [5,6,7],
                     "min_samples_split": [ 3,4,5],
                     "min_samples_leaf": [ 2,3, 4],
                     "bootstrap": [False],
                     "n_estimators": [90,100,110],
                     "criterion": ["gini"]}

    gsExtC = GridSearchCV(ExtC, param_grid=ex_param_grid, cv=kfold, scoring="accuracy", n_jobs=4, verbose=1)

    gsExtC.fit(train, y)

    ExtC_best = gsExtC.best_estimator_
    print(ExtC_best)
    # Best score
    print(gsExtC.best_score_)

def hyperparameterTuning(X_train, Y_train):
    kfold = StratifiedKFold(n_splits=10)
    DTC = DecisionTreeClassifier()

    adaDTC = AdaBoostClassifier(DTC, random_state=7)

    ada_param_grid = {"base_estimator__criterion": ["gini", "entropy"],
                      "base_estimator__splitter": ["best", "random"],
                      "algorithm": ["SAMME", "SAMME.R"],
                      "n_estimators": [1, 2,3,4,5],
                      "learning_rate": [0.0001, 0.001, 0.01, 0.1,0.15, 0.2,0.25, 0.3,0.35, 1.5]}

    gsadaDTC = GridSearchCV(adaDTC, param_grid=ada_param_grid, cv=kfold, scoring="accuracy", n_jobs=4, verbose=1)

    gsadaDTC.fit(X_train.values, Y_train.values)

    ada_best = gsadaDTC.best_estimator_
    print(gsadaDTC.best_score_)
    print(ada_best)

def getGradientBoostingTuned():
    return GradientBoostingClassifier(criterion='friedman_mse', init=None,
              learning_rate=0.25, loss='deviance', max_depth=4,
              max_features=0.45, max_leaf_nodes=None,
              min_impurity_decrease=0.0, min_impurity_split=None,
              min_samples_leaf=100, min_samples_split=2,
              min_weight_fraction_leaf=0.0, n_estimators=325,
              presort='auto', random_state=None, subsample=1.0, verbose=0,
              warm_start=False)


def gradientBoostingTuning(train, y):
    kfold = StratifiedKFold(n_splits=10)
    GBC = GradientBoostingClassifier()
    gb_param_grid = {'loss': ["deviance"],
                     'n_estimators': [312,325,337],
                     'learning_rate': [0.23,0.25,0.27],
                     'max_depth': [3,4,5],
                     'min_samples_leaf': [90,100,110],
                     'max_features': [0.5,0.45,0.4]
                     }

    gsGBC = GridSearchCV(GBC, param_grid=gb_param_grid, cv=kfold, scoring="accuracy", n_jobs=4, verbose=1)

    gsGBC.fit(train, y)

    GBC_best = gsGBC.best_estimator_
    print(GBC_best)
    # Best score
    print(gsGBC.best_score_)

def getSVCTuned():
    return SVC(C=3, cache_size=200, class_weight=None, coef0=0.0,
        decision_function_shape='ovr', degree=2, gamma=0.1, kernel='rbf',
        max_iter=-1, probability=True, random_state=None, shrinking=True,
        tol=0.001, verbose=False)

def SVCTuning(train, y):
    kfold = StratifiedKFold(n_splits=10)
    ### SVC classifier
    SVMC = SVC(probability=True)
    svc_param_grid = {'kernel': ['rbf'],
                      'gamma': [0.08, 0.01,0.12],
                      'C': [2,3,4],
                      'degree':[2,3,4]}

    gsSVMC = GridSearchCV(SVMC, param_grid=svc_param_grid, cv=kfold, scoring="accuracy", n_jobs=4, verbose=1)

    gsSVMC.fit(train, y)

    SVMC_best = gsSVMC.best_estimator_
    print(SVMC_best)
    # Best score
    print(gsSVMC.best_score_)

def getMLPTuned():
    return MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.8,
       beta_2=0.999, early_stopping=False, epsilon=1e-08,
       hidden_layer_sizes=(100,), learning_rate='constant',
       learning_rate_init=0.001, max_iter=300, momentum=0.9,
       nesterovs_momentum=True, power_t=0.4, random_state=2, shuffle=True,
       solver='adam', tol=0.0001, validation_fraction=0.1, verbose=False,
       warm_start=False)


def MLPTuning(train, y):
    kfold = StratifiedKFold(n_splits=10)
    ### SVC classifier
    MLP = MLPClassifier(random_state=2)
    mlp_param_grid = {'alpha': [0.0001],
                      'power_t': [0.4],
                      'max_iter': [400],
                      'beta_1':[0.8],
                      'beta_2': [0.999]}

    gsMLP = GridSearchCV(MLP, param_grid=mlp_param_grid, cv=kfold, scoring="accuracy", n_jobs=4, verbose=1)

    gsMLP.fit(train, y)

    MLP_best = gsMLP.best_estimator_
    print(MLP_best)
    # Best score
    print(gsMLP.best_score_)


def getDecisionTreeTuned():
    return DecisionTreeClassifier(random_state=2)

def DecisionTreeTuning(train, y):
    kfold = StratifiedKFold(n_splits=10)
    decisionTree = DecisionTreeClassifier(random_state=2)

    ## Search grid for optimal parameters
    dt_param_grid = {"max_depth": [None],
                     "max_features": [3,4,5],
                     "min_samples_split": [2, 3,4],
                     "min_samples_leaf": [1,2,3],
                     "min_weight_fraction_leaf":[0.0,0.01],
                     "min_impurity_decrease": [0.0,0.01],
                     }

    gsDT = GridSearchCV(decisionTree, param_grid=dt_param_grid, cv=kfold, scoring="accuracy", n_jobs=4, verbose=1)

    gsDT.fit(train, y)

    DT_best = gsDT.best_estimator_
    print(DT_best)
    # Best score
    print(gsDT.best_score_)

def getKNeighborsTuned():
    return KNeighborsClassifier(algorithm='auto', leaf_size=15, metric='minkowski',
           metric_params=None, n_jobs=1, n_neighbors=7, p=2,
           weights='uniform')

def KNeighborsTuning(train, y):
    kfold = StratifiedKFold(n_splits=10)
    KNeighbors = KNeighborsClassifier()

    ## Search grid for optimal parameters
    kn_param_grid = {"n_neighbors": [6,7,8],
                     "leaf_size": [15,20,25,30],
                     "p":[1,2,3]
                     }

    gsKN = GridSearchCV(KNeighbors, param_grid=kn_param_grid, cv=kfold, scoring="accuracy", n_jobs=4, verbose=1)

    gsKN.fit(train, y)

    KN_best = gsKN.best_estimator_
    print(KN_best)
    # Best score
    print(gsKN.best_score_)

def getLinearDiscriminantTuned():
    return LinearDiscriminantAnalysis()

def getLogisticRegressionTuned():
    return LogisticRegression(random_state=0, max_iter=5000, fit_intercept=False, tol=0.00001)

fileNameExtension = 'ABCEFGHI12'
fileNameExtensionTest = 'ABCEFGHI1'
train = pandas.read_csv("Data/Input/PreparedData/PreparedTrain_%s.csv" % (fileNameExtension),
                        index_col='PassengerId')
test = pandas.read_csv("Data/Input/PreparedData/PreparedTest_%s.csv" % (fileNameExtensionTest),
                       index_col='PassengerId')

features = ['Age',  'Fare','Parch', 'Sex', 'SibSp', 'Single',
            'SmallFamily', 'MediumFamily', 'LargeFamily', 'Embarked_C',
            'Embarked_Q', 'Embarked_S', 'Title_0', 'Title_1', 'Title_2', 'Title_3',
            'Pclass_1', 'Pclass_2', 'Pclass_3', 'Cabin_A', 'Cabin_B', 'Cabin_C',
            'Cabin_D', 'Cabin_E', 'Cabin_F', 'Cabin_G', 'Cabin_T', 'Cabin_X',  'Ticket_A', 'Ticket_A4', 'Ticket_A5', 'Ticket_AQ3', 'Ticket_AQ4',
                'Ticket_AS', 'Ticket_C', 'Ticket_CA', 'Ticket_CASOTON', 'Ticket_FC',
                'Ticket_FCC', 'Ticket_Fa', 'Ticket_LINE', 'Ticket_LP', 'Ticket_PC',
                'Ticket_PP', 'Ticket_PPP', 'Ticket_SC', 'Ticket_SCA3', 'Ticket_SCA4',
                'Ticket_SCAH', 'Ticket_SCOW', 'Ticket_SCPARIS', 'Ticket_SCParis',
                'Ticket_SOC', 'Ticket_SOP', 'Ticket_SOPP', 'Ticket_SOTONO2',
                'Ticket_SOTONOQ', 'Ticket_SP', 'Ticket_STONO', 'Ticket_STONO2',
                'Ticket_STONOQ', 'Ticket_SWPP', 'Ticket_WC', 'Ticket_WEP', 'Ticket_X']

y=train['Survived']
trainData = train[features]
testData = test[features]
start(trainData,testData,y)


