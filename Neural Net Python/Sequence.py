import pandas as pd
import math
import numpy as np

class Sequence:
    
    @staticmethod
    def convertToBinaryArray(array, bit_size):
        binArray = []
        for i in range(len(array)):
            binArray.append(Sequence.convertToBinary(array[i]))
        for i in range(len(binArray)):
            if len(binArray[i]) < bit_size:
                binArray[i] = [0]*(bit_size - len(binArray[i])) + binArray[i]
        return binArray
    
    @staticmethod
    def createWaveArray(awp_array, length, resolution):
        wave_array = []
        max_val = 0
        for i in range(length):
            wave_val = 0
            for n in range(len(awp_array)):
                curr_val = i*resolution
                wave_val += awp_array[n][0]*math.sin(awp_array[n][1]*curr_val + awp_array[n][2])
            if wave_val > max_val:
                max_val = wave_val
            wave_array.append(wave_val)
        scale_val = max_val * 2
        for i in range(len(wave_array)):
            wave_array[i] = max(0,(wave_array[i]/scale_val) + 0.5)
            
        return wave_array
    
    @staticmethod
    def getWaveVal(awp_array, length, resolution, val):
        wave_val = 0
        for n in range(len(awp_array)):
            wave_val += awp_array[n][0]*math.sin(awp_array[n][1]*val + awp_array[n][2])
            
        return wave_val
            
    
    @staticmethod
    def createMotionSensorBinArray(input_file, length, bit_size):
        df_sensor_events = pd.read_csv(input_file, header = 0, comment='#')
                                       
        sensor_list = df_sensor_events.ID.unique().tolist()
        sensor_list.sort()
        
        sensor_dict = {}
        for i in range(len(sensor_list)):
            sensor_dict[sensor_list[i]] = i
            
        formatted_data = []
        
        if length == -1:
            length = len(df_sensor_events)
        for i in range(length):
            sensor_id = sensor_dict[df_sensor_events.at[i, 'ID']]
            formatted_data.append(Sequence.convertToBinarySize(sensor_id, bit_size))
        
        return formatted_data
    
    @staticmethod
    def createMotionSensorTSet(input_file, length, look_back, bit_size):
        df_sensor_events = pd.read_csv(input_file, header = 0, comment='#')
                                       
        sensor_list = df_sensor_events.ID.unique().tolist()
        sensor_list.sort()
        
        sensor_dict = {}
        for i in range(len(sensor_list)):
            sensor_dict[sensor_list[i]] = i
            
        formatted_data = []
        
        if length == -1:
            length = len(df_sensor_events)
        for i in range(length):
            sensor_id = sensor_dict[df_sensor_events.at[i, 'ID']]
            formatted_data.append(sensor_id)
        
        return Sequence.createTrainingSet(formatted_data, look_back, bit_size)
    
    @staticmethod
    def getIOformat(index, array, look_back, bit_size):
        inputs = []
        for i in range(look_back):
            inputs += Sequence.convertToBinarySize(array[index + i], bit_size)
        outputs = Sequence.convertToBinarySize(array[index + look_back], bit_size)
        return[inputs,outputs]
        
    @staticmethod
    def getTrainingElemDecToBin(index, array, look_back, bit_size):
        inputs = []
        for i in range(look_back):
            inputs += Sequence.convertDecimalToBinaryArray(array[index + i], bit_size)
        outputs = Sequence.convertDecimalToBinaryArray(array[index + look_back], bit_size)
        return[inputs,outputs]
        
    @staticmethod
    def getIOformatBin(index, bin_array, look_back):
        inputs = []
        for i in range(look_back):
            inputs += bin_array[index + i]
        outputs = bin_array[index + look_back]
        return[inputs,outputs]
            
    @staticmethod
    def createTrainingSet(array, look_back, bit_size):
        training_set = []
        for i in range(len(array) - look_back):
            training_set.append(Sequence.getIOformat(i,array,look_back,bit_size))
        return training_set
        
    @staticmethod
    def convertToBinarySize(num, bitSize):
        binNum = Sequence.convertToBinary(num)
        if len(binNum) < bitSize:
            binNum = [0]*(bitSize - len(binNum)) + binNum
        return binNum
            
    @staticmethod
    def convertToBinary(num):
        if num <= 1:
            return [num]
        else:
            bit = num&1
            num >>=1 
            arr = Sequence.convertToBinary(num)
            arr.append(bit)
            return arr
        
    @staticmethod 
    def convertBinaryArrayToDecimal(binaryArray):
        bitSize = len(binaryArray)
        num = 0
        for i in range(bitSize):
            num += binaryArray[i] * (1 << ((bitSize-1) - i))
        num = num / ((1 << bitSize)-1)
        return num
    
    @staticmethod 
    def convertFuzzyBinaryArrayToDecimal(binaryArray):
        bitSize = len(binaryArray)
        num = 0
        for i in range(bitSize):
            val = int(round(binaryArray[i],0))
            num += val * (1 << ((bitSize-1) - i))
        num = num / ((1 << bitSize)-1)
        return num
    
    @staticmethod 
    def convertFuzzyBinaryArrayToInt(binaryArray):
        bitSize = len(binaryArray)
        num = 0
        for i in range(bitSize):
            val = int(round(binaryArray[i],0))
            num += val * (1 << ((bitSize-1) - i))
        return num
    
    @staticmethod
    def quantizeFuzzyBinaryArray(binaryArray):
        bitSize = len(binaryArray)
        q_binaryArray = []
        for i in range(bitSize):
            q_binaryArray.append(int(round(binaryArray[i],0)))
        return q_binaryArray
        
    @staticmethod
    def convertDecimalToBinaryArray(decimal, bitSize):
        maxVal = (1<<bitSize) - 1
        val = int(round(decimal*maxVal,0))
        return Sequence.convertToBinarySize(val, bitSize)
    
    @staticmethod
    def convertBinaryToOneHot(binary, length):
        num = Sequence.convertFuzzyBinaryArrayToInt(binary)
        return Sequence.createOneHot(num, length)
        
    @staticmethod
    def oneHotToBinary(one_hot):
        index = 0
        for i in range(len(one_hot)):
            if one_hot[i]  == 1:
                index = i
                break
        bit_size = Sequence.findBitSize(len(one_hot))
        return Sequence.convertToBinarySize(index, bit_size)
        
    @staticmethod
    def findBitSize(num):
        size = 1
        while(num > 1):
            num >>= 1
            size += 1
        return size
    
    @staticmethod
    def createSequence(seq_sample, num_iter):
        array = []
        for i in range(num_iter):
            array += seq_sample
        return array
    
    @staticmethod
    def netOutToBinProb(net_out, bin_val):
        prob = 1
        for i in range(len(net_out)):
            if bin_val[i] == 1:
                prob *= net_out[i]
            else:
                prob *= (1 - net_out[i])
        return prob
    
    @staticmethod
    def netOutToOneHotProb(net_out, one_hot):
        val = one_hot.index(max(one_hot))
        return net_out[val]
    
    @staticmethod 
    def createOneHot(val, num_dims):
        arr = [0] * num_dims
        arr[val] = 1
        return arr
    
    @staticmethod
    def convertFuzzyOneHot(one_hot):
        new_one_hot = [0]*len(one_hot)
        index = np.argmax(one_hot)
        new_one_hot[index] = 1
        return new_one_hot
                
    @staticmethod
    def compareOneHots(one_hot1, one_hot2):
        is_same = True
        for i in range(len(one_hot1)):
            if one_hot1[i] != one_hot2[i]:
                is_same = False
        return is_same
        