import pandas

class TitanicInstance:

    def __init__(self, train=None, test=None, features=None):
        if features is None:
            self.features = []
        else:
            self.features = features

        if train is None:
            self.train = pandas.Dataframe()
        else:
            self.train = train[features]

        if test is None:
            self.test = pandas.Dataframe()
        else:
            self.test = test[features]

        self.y=train['Survived']