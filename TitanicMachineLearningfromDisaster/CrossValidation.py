import numpy
import pandas
from TitanicMachineLearningfromDisaster import LogisticRegressionCS
from TitanicMachineLearningfromDisaster import TitanicYassineGhouzam
#cross validation of LogisticRegressionCS

def calcSubsamples(train, sizeOfSamples):
    subsamples = []
    permuted_indices = numpy.random.permutation(len(train))
    for i in range(sizeOfSamples):
        subsamples.append(train.iloc[permuted_indices[i::sizeOfSamples]])
    return subsamples

def validate(train, sizeOfSamples, classifier,y):

    subsamples=calcSubsamples(train, sizeOfSamples)

    for sample in subsamples:
        sample=sample.sort_index();
        trainSample = train.drop(sample.index)

        result = pandas.DataFrame(index=sample.index)
        yTrainSample=y.drop(sample.index)

        ytestCS = TitanicYassineGhouzam.regression(classifier, trainSample, sample, yTrainSample)

        ySample = y.drop(trainSample.index)

        TitanicYassineGhouzam.printScore(classifier, sample, ySample)
        #y = pandas.DataFrame(train['Survived'])
        TitanicYassineGhouzam.printScore(classifier, train, y)
        print()

def start():
    classifierCS = LogisticRegressionCS.LogisticRegressionCS(max_iter=1000, tolerance=0.00001);
    train = pandas.read_csv("PreparedTrain.csv", index_col='PassengerId')
    features = ['Age', 'Fare', 'Sex', 'Single', 'SmallFamily', 'MediumFamily', 'LargeFamily', 'EM_C', 'EM_Q', 'EM_S',
                'Title_0', 'Title_1', 'Title_2', 'Title_3', 'Pc_1', 'Pc_2', 'Pc_3']
    y = pandas.DataFrame(train['Survived'])

    validate(train[features], 10, classifierCS, y)
