import pickle

class SaveLoad:
    
    @staticmethod
    def saveObject(obj, filename):
        with open(filename, 'wb') as output:
            pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
    
    @staticmethod
    def loadObject(filename):
        with open(filename, 'rb') as input:
            obj = pickle.load(input)
            return obj