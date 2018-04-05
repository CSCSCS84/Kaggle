import pandas
import numpy
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations


def calcCorrelationOfClassifiers(result, classifiers):
    correlation = result[classifiers].corr().round(
        2)
    print(correlation)
    ax = sns.heatmap(correlation, annot=True)
    plt.show()
    return correlation

def calcLeastCorrelated(correlation, classifiers,num):
    combination=list(combinations(classifiers, num))
    print(len(combination))
    print(combination)
    smallestValue=10000;
    smallesCombination=None;
    for c in combination:
        correlation = result[list(c)].corr().round(
            3)
        sum=numpy.sum(correlation.values)
        #print(sum)
        if sum < smallestValue:
            smallesCombination=c
            smallestValue=sum
    print(smallesCombination)
    print(smallestValue)


result = pandas.read_csv("Data/Output/PredictedResultMultipleTrain.csv",
                         index_col='PassengerId')
classifiers = ['SVC', 'ExtraTreesClassifier', 'GradientBoostingClassifier', 'MLPClassifier',
               'LinearDiscriminantAnalysis', 'LogisticRegression', 'RandomForestClassifier', 'AdaBoostClassifier',
                'KNeighborsClassifier']
classifiersBest=['LinearDiscriminantAnalysis', 'LogisticRegression', 'AdaBoostClassifier', 'DecisionTreeClassifier', 'KNeighborsClassifier']

correlation=calcCorrelationOfClassifiers(result, classifiers)

calcLeastCorrelated(correlation,classifiers,5)