import numpy
import pandas
from TitanicMachineLearningfromDisaster import LogisticRegressionCS
from TitanicMachineLearningfromDisaster import TitanicYassineGhouzam
import seaborn as sns
import matplotlib.pyplot as plt

train = pandas.read_csv("PreparedTrain.csv", index_col='PassengerId')
features=['Age','Fare',  'Sex', 'Single', 'SmallFamily', 'MediumFamily', 'LargeFamily', 'EM_C', 'EM_Q', 'EM_S', 'Title_0', 'Title_1', 'Title_2', 'Title_3', 'Pc_1', 'Pc_2', 'Pc_3']


#features=['Age','Fare',  'Sex']


#g = sns.heatmap(train[features].corr(),annot=True, fmt = ".2f", cmap = "coolwarm")
#plt.show()
iterations=2;
for i in range(1,15):
    print("Iterations ")
    print(iterations)
    classifierCS=LogisticRegressionCS.LogisticRegressionCS(max_iter=iterations, tolerance=0.00001);
    result = pandas.DataFrame(index=train.index)
    y = pandas.DataFrame(train['Survived'])
    ytestCS = TitanicYassineGhouzam.regression(classifierCS, train[features], train[features], y)
    sampleSurvived = pandas.DataFrame(train['Survived'])
    TitanicYassineGhouzam.printScore(classifierCS, train[features], sampleSurvived)
    iterations*=2;
