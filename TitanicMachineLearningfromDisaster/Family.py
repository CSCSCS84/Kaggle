class Family:
    def __init__(self, name=None, familiysize=None, ticketnumber=None, numOfDead=None, numOfSurvived=None):
        if name is None:
            self.name = ''
        else:
            self.name = name

        if familiysize is None:
            self.familiysize = 0
        else:
            self.familiysize = familiysize

        if ticketnumber is None:
            self.ticketnumber = 0
        else:
            self.ticketnumber = ticketnumber

        if numOfDead is None:
            self.numOfDead = 0
        else:
            self.numOfDead = numOfDead
        if numOfSurvived is None:
            self.numOfSurvived = 0
        else:
            self.numOfSurvived = numOfSurvived
    def __hash__(self):
        return hash((self.name, self.familiysize, self.ticketnumber))

    def __eq__(self, obj):
        if (self.name == obj.name) and (self.familiysize == obj.familiysize) and (self.ticketnumber == obj.ticketnumber):
            return True
        else:
            return False

    def __str__(self):
        return "%s %s %s %s %s" % (self.name, self.familiysize, self.ticketnumber, self.numOfDead, self.numOfSurvived)
