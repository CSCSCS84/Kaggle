import numpy
import pandas
from sklearn.preprocessing import MinMaxScaler
from collections import Counter


class PrepareData:
    filenameExtension = []

    def scaleData(self, data, features):
        scaler = MinMaxScaler()
        data[features] = scaler.fit_transform(data[features])
        self.filenameExtension += '1'

    def dropOutliers(self, dataset, number, features):
        outliers = self.detectOutliers(dataset, number, features)
        train = dataset.drop(outliers)
        self.filenameExtension += '2'

    # detect outliers using Tuking method
    def detectOutliers(self, train, limitOfOutlierFeatures, features):
        outliers = [];
        for f in features:
            bounds = self.calculateBounds(train, f)
            outliersfeature = train[(train[f] < bounds[0]) | (train[f] > bounds[1])].index.values;
            outliers.extend(outliersfeature)

        outliersCount = Counter(outliers)
        multiple_outliers = list(k for k, v in outliersCount.items() if v > limitOfOutlierFeatures)
        return multiple_outliers

    def calculateBounds(self, train, feature):
        firstQuantil = train[feature].quantile(0.25, interpolation='linear')
        thirdQuantil = train[feature].quantile(0.75, interpolation='linear')
        IQR = thirdQuantil - firstQuantil
        lowerBound = firstQuantil - 1.5 * IQR
        upperBound = thirdQuantil + 1.5 * IQR
        return [lowerBound, upperBound]
