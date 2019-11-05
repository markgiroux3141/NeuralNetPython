import numpy
from Sequence import Sequence
import math

class Helper:
    
    @staticmethod
    def convertToBinaryData(raw_data):
        
        train_data = []
        test_data = []
        num_elem = []
        dictionary = []
        
        for i in range(0, len(raw_data), 2):
            train_data_elem = raw_data[i]
            test_data_elem = raw_data[i+1]
            train_data.append(train_data_elem)
            test_data.append(test_data_elem)
            train_unique = set(train_data_elem)
            test_unique = set(test_data_elem)
            total_unique = list(train_unique | test_unique)
            num_elem.append(len(total_unique))
            total_unique.sort()
            d = {}
            for n in range(len(total_unique)):
                d[total_unique[n]] = n
            dictionary.append(d)
        
        train_data_concat = []
        test_data_concat = []
        for i in range(len(train_data)):
            #for n in range(len(train_data[i])):
            for n in range(10):
                bit_size = Sequence.findBitSize(num_elem[i] - 1)
                new_elem = Sequence.convertToBinarySize(dictionary[i][train_data[i][n]],bit_size)
                if i > 0:
                    train_data_concat[n] += new_elem
                else:
                    train_data_concat.append(new_elem)
                    
            #for n in range(len(test_data[i])):
            for n in range(10):
                bit_size = Sequence.findBitSize(num_elem[i] - 1)
                new_elem = Sequence.convertToBinarySize(dictionary[i][test_data[i][n]],bit_size)
                if i > 0:
                    test_data_concat[n] += new_elem
                else:
                    test_data_concat.append(new_elem)
            
        return train_data_concat, test_data_concat
    
    @staticmethod
    def preProcessData(npzFile, array_indicies, values, rep_values):
        data =  numpy.load(npzFile)
        file_names = data.files
        raw_data = data[file_names[0]]
        for q in range(len(array_indicies)):
            for i in range(len(raw_data[array_indicies[q]])):
                for n in range(len(values)):
                    if raw_data[array_indicies[q]][i] == values[n]:
                        raw_data[array_indicies[q]][i] = rep_values[n]
                        break
        return raw_data
    
    @staticmethod
    def createlogScale(array, steps):
        array = array.astype(numpy.float)
        minVal = min(array)
        maxVal = max(array)
        minLog = math.log1p(minVal)
        maxLog = math.log1p(maxVal)
        logRange = maxLog - minLog
        stepSize = logRange/(steps)
        logArray = []
        for i in range(1, steps+1):
            val = math.exp(minLog + (stepSize * i)) - 1
            logArray.append(val)
        return logArray
    
    @staticmethod
    def getQuantizedValuesFromLogScale(array, logScale):
        array = array.astype(numpy.float)
        qArray = []
        for i in range(len(array)):
            for n in range(len(logScale)):
                if array[i] < logScale[n]:
                    qArray.append(n)
                    break
        return qArray
    
    @staticmethod
    def createCompositeInt(key, numArray):
        val = 0
        bitShift = 0
        for i in range(len(numArray)):
            val |= numArray[i] << bitShift
            bitShift += key[i]
        return val
        
    @staticmethod
    def decomposeCompositeInt(key, num):
        numArray = []
        bitShift = 0
        for i in range(len(key)):
            numArray.append((num&(((1<<key[i])-1)<< bitShift))>>bitShift)
            bitShift += key[i]
        return numArray
        
        
        
        
            
            
            
            
            
                