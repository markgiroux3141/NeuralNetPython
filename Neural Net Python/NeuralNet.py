import numpy as np
import math
import random
from enum import Enum

class Activation(Enum):
    NONE=0
    SIGMOID=1
    LINEAR=2
    RELU=3
    LEAKY_RELU=4
    TANH=5
    
class NeuralNet:
    
    def __init__(self,num_neurons_array, activation_array, learning_rate):
        self.num_layers = len(num_neurons_array)
        self.num_neurons_array = num_neurons_array
        self.activation_array = activation_array
        self.output_vals = []
        self.learning_rate = learning_rate
    
        self.weights = []
        self.weights_der = []
        self.bias = []
        self.bias_der = []
        self.neuron_delta = []
        self.neuron_val = []
        self.neuron_val.append([0] * num_neurons_array[0])
        self.neuron_delta.append([0] * num_neurons_array[0])
        for i in range(1, self.num_layers):
            neurons = []
            neurons_der = []
            bias_neurons = []
            bias_neurons_der = []
            neuron_delta = []
            neurons_vals = []
            for n in range(num_neurons_array[i]):
                weights = []
                weights_der = []
                bias_neurons.append(NeuralNet.rand())
                bias_neurons_der.append(0)
                neuron_delta.append(0)
                neurons_vals.append(0)
                for q in range(num_neurons_array[i-1]):
                    weights.append(NeuralNet.rand())
                    weights_der.append(0)
                neurons.append(weights)
                neurons_der.append(weights_der)
            self.weights.append(neurons)
            self.weights_der.append(neurons_der)
            self.bias.append(bias_neurons)
            self.bias_der.append(bias_neurons_der)
            self.neuron_delta.append(neuron_delta)
            self.neuron_val.append(neurons_vals)
            
    def run(self, input_vals):
        for i in range(self.num_neurons_array[0]):
            self.neuron_val[0][i] = input_vals[i]
        for i in range(1, self.num_layers):
            for n in range(self.num_neurons_array[i]):
                sum = 0
                for q in range(len(self.weights[i - 1][n])):
                    sum += self.neuron_val[i - 1][q] * self.weights[i-1][n][q]
                self.neuron_val[i][n] = sum + self.bias[i-1][n]
                self.neuron_val[i][n] = NeuralNet.activationFunction(self.neuron_val[i][n], self.activation_array[i])
        self.output_vals = self.neuron_val[self.num_layers - 1] 
        
    def back_prop(self, train_data):
        NeuralNet.setNeuronsZero(self.neuron_delta)
        for i in range(self.num_layers - 1, 0, -1):
            if i == self.num_layers - 1:
                for n in range(self.num_neurons_array[i]):
                    o = self.output_vals[n]
                    t = train_data[n]
                    error_net_der = (o-t)*NeuralNet.derActivationFunction(o,self.activation_array[i])
                    self.bias_der[i-1][n] = error_net_der
                    for q in range(self.num_neurons_array[i-1]):
                        weight_der = error_net_der*self.neuron_val[i-1][q]
                        self.neuron_delta[i-1][q] += error_net_der*self.weights[i-1][n][q]
                        self.weights_der[i-1][n][q] = weight_der
            else:
                for n in range(self.num_neurons_array[i]):
                    error_net_der = self.neuron_delta[i][n] * NeuralNet.derActivationFunction(self.neuron_val[i][n], self.activation_array[i])
                    self.bias_der[i-1][n] = error_net_der
                    for q in range(self.num_neurons_array[i-1]):
                        weight_der = error_net_der*self.neuron_val[i-1][q]
                        self.neuron_delta[i-1][q] += error_net_der*self.weights[i-1][n][q]
                        self.weights_der[i-1][n][q] = weight_der
        for i in range(0,self.num_layers - 1):
            for n in range(self.num_neurons_array[i+1]):
                self.bias[i][n] -= self.bias_der[i][n] * self.learning_rate
                for q in range(self.num_neurons_array[i]):
                    self.weights[i][n][q] -= self.weights_der[i][n][q] * self.learning_rate    
        
    @staticmethod            
    def activationFunction(val, activation):
        if activation == Activation.SIGMOID:
            return NeuralNet.sigmoid(val)
        elif activation == Activation.LINEAR:
            return val
        
    @staticmethod
    def derActivationFunction(val, activation):
        if activation == Activation.SIGMOID:
            return val*(1-val)
        if activation == Activation.LINEAR:
            return 1
       
    @staticmethod             
    def rand():
        return (random.random() * 2) - 1
    
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + math.exp(-x))
                    
    @staticmethod
    def setNeuronsZero(neurons):
        for i in range(len(neurons)):
            for n in range(len(neurons[i])):
                neurons[i][n] = 0
                
    def displayNet(self):
        print('weights ', self.weights)
        print('biases ', self.bias)
    