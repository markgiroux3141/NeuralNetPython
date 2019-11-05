from NeuralNet import NeuralNet, Activation
from Sequence import Sequence
from Helper import Helper
from Hebbian import Hebbian
from SaveLoad import SaveLoad
import numpy as np
import matplotlib.pyplot as plt
from formatDataClass import FormatDataClass
from intervalClass import IntervalClass
import math
import random
import numpy as np
import sys

# XOR GATE
"""
neural_net = NeuralNet([2,3,1],[Activation.NONE, Activation.SIGMOID, Activation.LINEAR], 0.1)

xor_inputs = [[0,0],[0,1],[1,0],[1,1]]
xor_outputs = [[0],[1],[1],[0]]

epochs = 5000

for i in range(epochs):
    error = 0
    for n in range(len(xor_inputs)):
        neural_net.run(xor_inputs[n])
        neural_net.back_prop(xor_outputs[n])
        o = neural_net.output_vals[0]
        t = xor_outputs[n][0]
        error = 0.5*(t-o)*(t-o)
    if i%1000 == 0:
        print("epoch ", i, " error ", error)
        
    
for i in range(len(xor_inputs)):
    neural_net.run(xor_inputs[i])
    print(neural_net.output_vals)
"""
#LEARN PATTERN
"""
neural_net = NeuralNet([3,10,3])

index = 0

outputs = [[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]]

for i in range(30000):
    if i%5000 == 0:
        print("epoch ", i)
    neural_net.run(outputs[index])
    neural_net.back_prop(outputs[(index + 1) % len(outputs)])
    index += 1
    if index >= len(outputs):
        index = 0

for i in range(len(outputs)):
    neural_net.run(outputs[i])
    print(neural_net.output_vals)
"""

#LEARN PATTERN WITH LOOKBACK
"""
neural_net = NeuralNet([6,10,3])

inputs = [[0,0,0,0,0,1],[0,0,1,0,1,0],[0,1,0,0,1,1],[0,1,1,1,0,0],[1,0,0,1,0,1],[1,0,1,1,1,0],[1,1,0,1,1,1],[1,1,1,0,0,0]]
outputs = [[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1],[0,0,0],[0,0,1]]

for i in range(30000):
    error = 0
    for n in range(len(inputs)):
        neural_net.run(inputs[n])
        neural_net.back_prop(outputs[n])
        for q in range(len(outputs[n])):
            o = neural_net.output_vals[q]
            t = outputs[n][q]
            error += 0.5 * ((t-o)*(t-o))
    if i%1000 == 0:
        print("epoch ", i, " error ", error)

for i in range(len(inputs)):
    neural_net.run(inputs[i])
    print(neural_net.output_vals)

"""
#MOTION SENSOR DATA
    
"""

look_back = 10
training_data_length = -1
bit_size = 4
num_epochs = 100

neural_net = NeuralNet([bit_size*look_back, 10, 10, bit_size])

input_file = 'aruba1-reduced-sensors-training-Dec2010.csv'
bin_array = Sequence.createMotionSensorBinArray(input_file, training_data_length, bit_size)

for i in range(num_epochs):
    for n in range(len(bin_array) - look_back):
        error = 0
        training_elem = Sequence.getIOformatBin(n, bin_array, look_back)
        neural_net.run(training_elem[0])
        neural_net.back_prop(training_elem[1])
        for q in range(len(training_elem[1])):
            o = neural_net.output_vals[q]
            t = training_elem[1][q]
            error += 0.5 * ((t-o)*(t-o))
        
        if n%1000 == 0:
            print("epoch ", i, "training element ", n, " error ", error)
        
"""

#MOTION SENSORS

input_file = 'tulum2-reduced-sensors-training-Dec2009-Feb2010.csv'
test_file = 'tulum2-reduced-sensors-test-Mar2010.csv'
sensor_list, sensor_dict, state_list, state_dict = FormatDataClass.getUniqueSensorListAndDict(test_file)

q_steps = 8
max_hours = 24
#key = [Sequence.findBitSize(len(sensor_list)-1), Sequence.findBitSize(max_hours - 1), Sequence.findBitSize(q_steps - 1), Sequence.findBitSize(len(state_list) - 1)]
key = [len(sensor_list), max_hours, q_steps, len(state_list)]
prob_thresh = 0.02
look_back = 1

state_list = ['ON','OPEN','CLOSE']
state_dict = {'ON':0, 'OPEN':1, 'CLOSE':2}

formatted_data = FormatDataClass.formatData(input_file, state_list, sensor_list)
formatted_data_test = FormatDataClass.formatData(test_file, state_list, sensor_list)

anomaly_ind, anomaly_list = FormatDataClass.getAnomaliesFromFormattedData(formatted_data_test)
anomaly_set = set(anomaly_list)

print("Data Formatted")

ln_formatted_data = FormatDataClass.logNormalizeInterval(formatted_data)
ln_formatted_data_test = FormatDataClass.logNormalizeInterval(formatted_data_test)

log_interval_scale = Helper.createlogScale(np.array(ln_formatted_data)[:,1], q_steps)

quant_intervals = Helper.getQuantizedValuesFromLogScale(np.array(ln_formatted_data)[:,1], log_interval_scale)
quant_intervals_test = Helper.getQuantizedValuesFromLogScale(np.array(ln_formatted_data_test)[:,1], log_interval_scale)

for i in range(len(formatted_data)-1):
    formatted_data[i][1] = quant_intervals[i]
    
for i in range(len(formatted_data_test)-1):
    formatted_data_test[i][1] = quant_intervals_test[i]
    
data_for_net = FormatDataClass.formatForNetOneHot(formatted_data, sensor_dict, state_dict, key)
data_for_net_test = FormatDataClass.formatForNetOneHot(formatted_data_test, sensor_dict, state_dict, key)

#data_train = np.array(data_for_net)
#data_test = np.array(data_for_net_test)

#np.save("aruba_train_binary", data_train)
#np.save("aruba_test_binary", data_test)
#sys.exit()
print("Data Formatted For Net")

train_data = FormatDataClass.createTrainData(data_for_net, look_back, key, len(sensor_list))
test_data = FormatDataClass.createTrainData(data_for_net_test, look_back, key, len(sensor_list))
print("Train, Test Data Formatted")

num_epochs = 10
input_size = sum(key)
output_size = len(sensor_list)
learning_rate = 0.1
"""
neural_net = NeuralNet([input_size * look_back,40,40,output_size],[Activation.NONE, Activation.SIGMOID, Activation.SIGMOID, Activation.SIGMOID, Activation.SIGMOID], learning_rate)   

for i in range(num_epochs):
    for n in range(len(train_data)):
        rand_elem = random.randint(0,len(train_data)-1)
        error = 0
        neural_net.run(train_data[rand_elem][0])
        neural_net.back_prop(train_data[rand_elem][1])
        for q in range(len(train_data[rand_elem][1])):
            o = neural_net.output_vals[q]
            t = train_data[rand_elem][1][q]
            error += 0.5 * ((t-o)*(t-o))
        if n%1000 == 0:
            print("epoch ", i, "training element ", n, " error ", error)

SaveLoad.saveObject(neural_net, 'net.net')
print("Neural Net Saved")
"""
neural_net = SaveLoad.loadObject('net.net')
print("Neural Net Loaded")

anomaly_array = []
false_positive_array = []
X = []
start_point = 200
end_point = 2000
#for q in range(start_point,end_point,20):
num_anomaly = 0
num_fp = 0
num_tp = 0
anomaly_count = 0
num_normal = 0
anomalies = []
total_anomalies = []
#anomaly_label = []
#prob_thresh = 1/q
for i in range(look_back,len(test_data)):
    neural_net.run(test_data[i-look_back][0])
    prediction = neural_net.output_vals
    actual = test_data[i-look_back][1]
    prob_of_sensor = Sequence.netOutToOneHotProb(prediction, actual)
    if i in anomaly_ind:
        #anomaly_label.append([formatted_data_test[i][2]])
        print(formatted_data_test[i][2], " ", prob_of_sensor)
        total_anomalies.append(formatted_data_test[i][2])
        anomaly_count += 1
    if prob_of_sensor <= prob_thresh:
        num_anomaly += 1
        s = str(formatted_data_test[i][2])
        if s.__contains__('test'):
            num_tp += 1
            anomalies.append(formatted_data_test[i][2])
        else:
            num_fp += 1
    else:
        num_normal += 1
    if i % 1000 == 0:
        print("Iteration ", q, " Processing ", i, " entries done")
        
print("Anomaly Count ", num_anomaly)
print("Normal Count ", num_normal)
print("Anomalies ", anomalies)
print("True Positives ", num_tp)
print("False Positives ", num_fp)

#a = np.array(anomaly_ind)
#b = np.array(anomaly_label)
#np.save("anomaly_index", a)
#np.save("anomaly_label", b)

#print(a)
#print(b)
#sys.exit()

num_tn = len(test_data) - (num_fp + anomaly_count)
recall = num_tp/anomaly_count
precision = num_tp/(num_tp+num_fp)
true_negative_rate = num_tn/(num_tn+num_fp)
print("Recall Accuracy ", recall*100, "%")
print("Precision Accuracy ", precision*100,"%")
print("True Negative Accuracy ", true_negative_rate*100,"%")
print("Adjusted Accuracy ", ((recall + true_negative_rate)/2)*100,"%")

distinct_anomalies_caught = set(anomalies)
distinct_anomalies_total = set(total_anomalies)
d_tp = len(distinct_anomalies_caught)
d_fn = len(distinct_anomalies_total) - d_tp
distinct_recall_accuracy = d_tp/(d_tp+d_fn)
print("Distinct Anomaly Recall Accuracy ", distinct_recall_accuracy*100,"%")
print("Adjusted Distinct Anomaly Accuracy ", ((true_negative_rate+distinct_recall_accuracy)/2)*100,"%")

"""
X.append(1/q)
anomaly_array.append(len(set(anomalies)))
false_positive_array.append(num_fp)

plt.xlim(1/start_point, 1/end_point)
plt.plot(X,anomaly_array)
plt.savefig('anomaly_graph')
plt.show()

plt.xlim(1/start_point, 1/end_point)
plt.plot(X,false_positive_array)
plt.savefig('false_positive_graph')
plt.show()
"""


"""
num_sample_points = 50
last_vals = []
points_Y = []
points_X = []
for i in range(look_back):
    last_vals.append(Sequence.convertToBinarySize(data_for_net[i],net_output_bitSize))
    points_X.append(i)
    points_Y.append(data_for_net[i])
    
for i in range(0, num_sample_points):
    input_array = []
    for n in range(look_back):
        input_array += last_vals[n]
    neural_net.run(input_array)
    new_val_bin_array = neural_net.output_vals
    new_val = Sequence.convertFuzzyBinaryArrayToInt(new_val_bin_array)
    print("value ", new_val, " output ", new_val_bin_array)
    points_X.append(i + look_back)
    points_Y.append(new_val)
    for n in range(look_back - 1):
        last_vals[n] = last_vals[n+1]
    last_vals[look_back-1] = Sequence.quantizeFuzzyBinaryArray(new_val_bin_array)

plt.plot(points_X, points_Y)
plt.show()  
"""

"""
numVals = len(sensor_list)
for i in range(numVals):
    inputVal = Sequence.convertToBinarySize(i, net_output_bitSize)
    neural_net.run(inputVal)
    print("output for sensor ", sensor_list[i], " = ", neural_net.output_vals)
    sensorVals = ""
    for n in range(numVals):
        prob = Sequence.netOutToBinProb(neural_net.output_vals, n)
        sensorVals += (" " + sensor_list[n] + " -> " + str(round((prob*100),2)) + "%")
    print("          ", sensorVals)
    print(" ")
"""

#FORMAT DATA INTO CONCAT BINARY
"""
raw_data = Helper.preProcessData('neural_network_features_aruba_2.npz', [2,3], ['ONM026', 'OF'], ['ON', 'OFF'])

train, test = Helper.convertToBinaryData(raw_data)

np.savez_compressed('neural_network_features_aruba_2_binary2.npz',[train, test],['train_vectors','test_vectors'])
print("Done")


data = np.load('neural_network_features_aruba_2_binary2.npz')
file_names = data.files
raw_data = data[file_names[0]]
for i in range(10000):
    sensor_vec = raw_data[0][i][-6:]
    print(Sequence.convertFuzzyBinaryArrayToInt(sensor_vec))


key = [4,5,1,3]
numArray = [5,26,0,6]
val = Helper.createCompositeInt(key,numArray)
print(val)
print(Helper.decomposeCompositeInt(key,val))
print(Sequence.convertToBinary(val))
"""
"""

#HEBBIAN

hebbian = Hebbian()

hebbian.createData(100,-10,10,2,0.3)

hebbian.train()
"""

# FUNCTION SHAPE
"""
def wave(x):
    return (Sequence.getWaveVal([[.6,10,0],[.2,25,.5]],100,0.01,x)+1)/2

neural_net = NeuralNet([1,20,20,1],[Activation.NONE, Activation.SIGMOID, Activation.SIGMOID, Activation.LINEAR], 0.1)

epochs = 10000
array_length = 100
array_resolution = 0.01
output_array = []
target_array = []
xVals = []
xShuffle = []
for i in range(array_length):
    xShuffle.append(i*array_resolution)


for q in range(epochs):
    random.shuffle(xShuffle)
    for i in range(array_length):
        error = 0
        inputVal = xShuffle[i]
        neural_net.run([inputVal])
        out = neural_net.output_vals[0]
        target = wave(inputVal)
        neural_net.back_prop([target])
        error = 0.5 * (target - out) * (target - out)
    if q % 10 == 0:
        print("epoch ", q, " error ", error)


for i in range(array_length):
    inputVal = i * array_resolution
    neural_net.run([inputVal])
    out = neural_net.output_vals[0]
    output_array.append(out)
    target_array.append(wave(inputVal))
    xVals.append(inputVal)
    
plt.plot(xVals, output_array)
plt.plot(xVals, target_array)
plt.show()  

"""


