from NeuralNet import NeuralNet
import random
import numpy as np
import matplotlib.pyplot as plt

class Hebbian:
    
    def __init__(self):
        self.data = []
        self.labels = []
        self.neuralNet = NeuralNet([2,10,1])
        self.num_epochs = 100
        
    def train(self):
        error = 0
        Hebbian.shuffleArray(self.data)
        
        outArray = []
        
        for q in range(len(self.data)):
            self.neuralNet.run(self.data[q])
            o = self.neuralNet.output_vals[0]
            outArray.append(o)
        cutoff = Hebbian.findCutoff(outArray)
        
        """
        for i in range(self.num_epochs):
            for n in range(len(self.data)):
                inputVal = self.data[n]
                self.neuralNet.run(inputVal)
                outp = self.neuralNet.output_vals
                trainVal = Hebbian.returnIOfromCutoff(outp[0], cutoff)
                self.neuralNet.back_prop([trainVal])
                error = 0.5 * ((trainVal-outp[0])*(trainVal-outp[0]))
            if i%10 == 0:
                print("epoch ", i, " error ", error)
            """
            
        data = np.array(self.data)
        
        for i in range(len(self.data)):
            self.neuralNet.run(self.data[i])
            o = self.neuralNet.output_vals
            self.labels[i] = Hebbian.returnIOfromCutoff(o[0],cutoff)
            #self.labels[i] = int(round(o[0],0))

        plt.scatter(data[:,0], data[:,1], c=self.labels, alpha=1)
        plt.title('Hebbian Test')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()
        
    
    def createData(self, num_points, minVal, maxVal, num_centers, spread):
        points = []
        for i in range(num_centers):
            center = [Hebbian.randRange(minVal,maxVal), Hebbian.randRange(minVal,maxVal)]
            x = np.random.normal(center[0], spread, num_points)
            y = np.random.normal(center[1], spread, num_points)
            for n in range(num_points):
                points.append([x[n],y[n]])
                self.labels.append(i)
        self.data = points
        
    @staticmethod             
    def rand():
        return (random.random() * 2) - 1
    
    @staticmethod
    def randRange(min, max):
        return (random.random() * (max-min)) + min
    
    @staticmethod
    def shuffleArray(array):
        random.shuffle(array)
        
    @staticmethod
    def findCutoff(array):
        minVal = min(array)
        maxVal = max(array)
        return (minVal + maxVal)/2
    
    @staticmethod
    def returnIOfromCutoff(val, cutoff):
        if val < cutoff:
            return 0
        else:
            return 1
    
        