import numpy
import pandas
from sklearn.preprocessing import MinMaxScaler
from collections import Counter

def scaleData(train_data,featuresToScale):
    #print(train_data)
    scaler = MinMaxScaler()

    train_data[featuresToScale] = scaler.fit_transform(train_data[featuresToScale])
    return train_data;

#detect outliners using Tuking method
def detectOutliners(train,numOfOutlines,features):
    outliners=[];
    firstQuantil=train.quantile(0.25)
    thirdQuantil=train.quantile(0.75)
    IQR=thirdQuantil-firstQuantil;
    lower=firstQuantil-1.5*IQR;
    upper=thirdQuantil+1.5*IQR;

    outliers=[];
    for f in features:
        firstQuantil = train[f].quantile(0.25,interpolation='linear')
        thirdQuantil = train[f].quantile(0.75,interpolation='linear')
        IQR = thirdQuantil - firstQuantil;
        lower = firstQuantil - 1.5 * IQR;
        upper = thirdQuantil + 1.5 * IQR;
        outliersfeature=train[(train[f]<lower) | (train[f]>upper)].index.values;
        outliers.extend(outliersfeature)
    outliersCount=Counter(outliers)

    multiple_outliers= list( k for k, v in outliersCount.items() if v > numOfOutlines )
    return multiple_outliers