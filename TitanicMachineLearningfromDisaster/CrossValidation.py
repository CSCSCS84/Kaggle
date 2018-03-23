import numpy
import pandas

#cross validation using k-fold Method

def validate(train, sizeOfSamples, classifier, y):
    subsamples=calculateSubsamples(train, sizeOfSamples)
    scores = []
    for sample in subsamples:
        sample = sample.sort_index();
        trainData = train.drop(sample.index)
        yTrainData = y.drop(sample.index)
        classifier.fit(trainData, yTrainData)
        ySample = y.drop(trainData.index)
        score = classifier.score(sample, ySample)
        scores.append(score)
    return scores


def calculateSubsamples(train, sizeOfSamples):
    subsamples = []
    permuted_indices = numpy.random.permutation(len(train))
    for i in range(sizeOfSamples):
        subsamples.append(train.iloc[permuted_indices[i::sizeOfSamples]])
    return subsamples


