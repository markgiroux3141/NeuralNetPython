import pandas as pd
from datetime import date, time, datetime, timedelta
from intervalClass import IntervalClass
import numpy as np
from Helper import Helper
from Sequence import Sequence
import math
import sys

class FormatDataClass:
    
    @staticmethod
    def getUniqueSensorListAndDict(input_file):
        df_sensor_events = pd.read_csv(input_file, header = 0, comment='#')
        sensor_list = df_sensor_events.ID.unique().tolist()
        sensor_list.sort()
        
        state_list = df_sensor_events.Value.unique().tolist()
        state_list.sort()
                
        sensor_dict = {}
        for i in range(len(sensor_list)):
            sensor_dict[sensor_list[i]] = i
            
        state_dict = {}
        for i in range(len(state_list)):
            state_dict[state_list[i]] = i
            
        return sensor_list, sensor_dict, state_list, state_dict
    
    @staticmethod
    def formatData(input_file, state_list, sensor_list):
        df_sensor_events = pd.read_csv(input_file, header = 0, comment='#')
        formattedData = []
        
        lastTime = 0
        for i in range(len(df_sensor_events)):
            if df_sensor_events.at[i, 'Value'] in state_list and df_sensor_events.at[i, 'ID'] in sensor_list:
                strTime =  df_sensor_events.at[i, 'Date'] + ' ' + df_sensor_events.at[i, 'Time']
                if i == 0:
                    lastTime = strTime
                else:
                    interval = IntervalClass.calculateInterval(lastTime,  strTime)
                    hour = IntervalClass.getHourFromDTString(strTime)
                    formattedData.append([df_sensor_events.at[i,'ID'], float(interval), df_sensor_events.at[i, 'Activity'], int(hour), df_sensor_events.at[i, 'Value']])
                    lastTime = strTime
        return formattedData
            
    @staticmethod
    def quantizeInterval(formatted_data, q_steps):
        interval_array = []
        len_f_data = len(formatted_data)
        for i in range(len_f_data):
            interval_array.append(float(formatted_data[i][1]))
        log_scale = Helper.createlogScale(interval_array, q_steps)
        quant_vals = Helper.getQuantizedValuesFromLogScale(interval_array, log_scale)
        for i in range(len_f_data):
            formatted_data[i][1] = quant_vals[i]
        return formatted_data
    
    @staticmethod
    def logNormalizeInterval(formatted_data):
        max_val = 0
        for i in range(len(formatted_data)):
            val = math.log1p(formatted_data[i][1])
            formatted_data[i][1] = val
            if val > max_val:
                max_val = val
        for i in range(len(formatted_data)):
            formatted_data[i][1] /= max_val
        return formatted_data
    
    @staticmethod
    def shiftInterval(formatted_data):
        for i in range(len(formatted_data)-1):
            formatted_data[i][1] = formatted_data[i+1][1]
        
    @staticmethod
    def formatForNet(q_formatted_data, sensor_dict, key):
        data_for_net = []
        for i in range(len(q_formatted_data)):
            sensor_id = sensor_dict[q_formatted_data[i][0]]
            #sensor_interval = q_formatted_data[i][1]
            #combined_val = Helper.createCompositeInt(key, [sensor_id, sensor_interval])
            data_for_net.append(sensor_id)
        return data_for_net
    
    @staticmethod
    def createTrainData(formatted_data_net, look_back, key, one_hot_length):
        formatted_data_test_train = []
        for i in range(len(formatted_data_net) - look_back):
            train = []
            for n in range(look_back):
                train += formatted_data_net[i+n]
            #test = Sequence.convertBinaryToOneHot(formatted_data_net[i+look_back][:key[0]], one_hot_length)
            test = formatted_data_net[i+look_back][:key[0]]
            formatted_data_test_train.append([train, test])
        return formatted_data_test_train
    
    @staticmethod
    def formatForNetOneHot(formatted_data, sensor_dict, num_sensors):
        formatted_data_net = []
        for i in range(len(formatted_data)):
            sensor_id = sensor_dict[formatted_data[i][0]]
            one_hot = Sequence.createOneHot(sensor_id, num_sensors)
            formatted_data_net.append(one_hot + [formatted_data[i][1]])
        return formatted_data_net
    
    @staticmethod
    def formatForNetBinary(formatted_data, sensor_dict, state_dict, key):
        formatted_data_net = []
        for i in range(len(formatted_data)):
            sensor_bit_size = key[0]
            hour_bit_size = key[1]
            interval_bit_size = key[2]
            state_bit_size = key[3]
            sensor_id = sensor_dict[formatted_data[i][0]]
            state_id = state_dict[formatted_data[i][4]]
            binary_sensor_id = Sequence.convertToBinarySize(sensor_id, sensor_bit_size)
            binary_hours = Sequence.convertToBinarySize(formatted_data[i][3], hour_bit_size)
            binary_interval = Sequence.convertToBinarySize(formatted_data[i][1], interval_bit_size)
            binary_state = Sequence.convertToBinarySize(state_id, state_bit_size)
            formatted_data_net.append(binary_sensor_id + binary_hours + binary_interval + binary_state)
        return formatted_data_net
    
    @staticmethod
    def formatForNetOneHot(formatted_data, sensor_dict, state_dict, key):
        formatted_data_net = []
        for i in range(len(formatted_data)):
            sensor_size = key[0]
            hour_size = key[1]
            interval_size = key[2]
            state_size = key[3]
            sensor_id = sensor_dict[formatted_data[i][0]]
            state_id = state_dict[formatted_data[i][4]]
            one_hot_sensor_id = Sequence.createOneHot(sensor_id, sensor_size)
            one_hot_hours = Sequence.createOneHot(formatted_data[i][3], hour_size)
            one_hot_interval = Sequence.createOneHot(int(formatted_data[i][1]), interval_size)
            one_hot_state = Sequence.createOneHot(state_id, state_size)
            formatted_data_net.append(one_hot_sensor_id + one_hot_hours + one_hot_interval + one_hot_state)
        return formatted_data_net
        
        
    @staticmethod
    def formatDataTimeOfDay(input_file, sensor_list, state_list, startTime, endTime):
        df_sensor_events = pd.read_csv(input_file, header = 0, comment='#')
        formattedData1 = []
        formattedData2 = []
        
        for i in range(len(df_sensor_events)):
            t = IntervalClass.stringToTime(str(df_sensor_events.at[i, 'Time']))
            hour = t.hour
            if df_sensor_events.at[i, 'ID'] in sensor_list and df_sensor_events.at[i, 'Value'] in state_list:
                array_item = [df_sensor_events.at[i,'ID'], df_sensor_events.at[i, 'Date'] + ' ' + df_sensor_events.at[i, 'Time'], df_sensor_events.at[i, 'Activity']]
                if startTime < endTime:
                    if hour >= startTime and hour < endTime:
                        formattedData1.append(array_item)
                    else:
                        formattedData2.append(array_item)
                else:
                    if hour >= startTime or hour < endTime:
                        formattedData1.append(array_item)
                    else:
                        formattedData2.append(array_item)
                        
        return formattedData1, formattedData2
    
    @staticmethod
    def getAnomaliesFromFormattedData(formatted_data):
        anomaly_ind = []
        anomaly_list = []
        for i in range(len(formatted_data)):
            s = str(formatted_data[i][2])
            if s.__contains__("test"):
                anomaly_ind.append(i)
                anomaly_list.append(formatted_data[i][2])
        return anomaly_ind, anomaly_list