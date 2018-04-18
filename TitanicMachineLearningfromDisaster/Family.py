class Family:
    def __init__(self, name=None, familiysize=None, ticket=None, numOfDead=None, numOfSurvived=None):
        if name is None:
            self.name = ''
        else:
            self.name = name

        if familiysize is None:
            self.familiysize = 0
        else:
            self.familiysize = familiysize

        if ticket is None:
            self.ticket = ''
        else:
            self.ticket = ticket

        if numOfDead is None:
            self.numOfDead = 0
        else:
            self.numOfDead = numOfDead
        if numOfSurvived is None:
            self.numOfSurvived = 0
        else:
            self.numOfSurvived = numOfSurvived
    def __hash__(self):
        return hash((self.name, self.familiysize, self.ticket))

    def __eq__(self, obj):
        if (self.name == obj.name) & (self.familiysize == obj.familiysize) & (self.ticket == obj.ticket):
            return True
        else:
            return False

    def __str__(self):
        return "%s %s %s %s %s" % (self.name, self.familiysize, self.ticket, self.numOfDead, self.numOfSurvived)
