from functions import *
import numpy as np
import math

# used http://neuralnetworksanddeeplearning.com/chap2.html
# and https://www.3blue1brown.com/lessons/backpropagation-calculus





class Network(object):

    def gen_biases(self, layers, layer):
        # initalize a bias matrix (meant to be overriden)
        # should be an np array of size `layers[layer]`
        # layer is the index the weight comes from
        # layers is all layers

        return np.random.default_rng().standard_normal(size=(layers[layer], 1))
     
    def gen_weights(self, layers, layer):
        # intialize a weight matrix (same as above)
        # should return a matrix of size 
        # `(layers[i], layers[i-1])`
        

        return np.random.default_rng().standard_normal(size=(layers[layer], layers[layer-1]))

    def __init__(self, inputLayers, activationFunction, derivFunction):
        # layers should be [(size1), (size2), etc.]
        # eg. [1, 2, 3]
        # 1 input, 2 hidden layer, 3 output

        # activationFunction should take in one number x
        # and return some output number. derivFunction should
        # be that function's derivative

        self.function = np.vectorize(activationFunction, otypes=[float])
        self.derivative = np.vectorize(derivFunction, otypes=[float])


        self.weights = []
        self.biases = []



        for i in range(1, len(inputLayers)):

            self.biases.append(self.gen_biases(inputLayers, i))
            # list as long as inputLayers[i], filled with
            # normal distribution mean 0 std deviation 1


            self.weights.append(self.gen_weights(inputLayers, i))
            # weights in scheme jk, from neuron j at prev layer
            # to neuron k at next layer

    def forward_pass(self, inputs):
    # inputs should be a list with size equal to size of
    # the input layer


        activations = np.array(inputs)


        for (weights, bias) in zip(self.weights, self.biases):
            activations = np.matmul(weights, activations)
            activations = np.add(activations, bias)
            activations = self.function(activations)

        return activations
    
    def full_forward_pass(self, inputs):
    # inputs should be same as forward_pass

    # returns (zActivations, activations) for entire network, zActivations and activations are
    # lists of np arrays

        activations = [inputs]
        zActivations = []

        for (weights, biases) in zip(self.weights, self.biases):
            z = np.matmul(weights, activations[-1])
            z = np.add(z, biases)
            zActivations.append(z)
            activation = self.function(z)
            activations.append(activation)

        return (zActivations, activations)
    
    def gradient_descent(self, inputs, learning_rate, epochs):
        # inputs should be [[(list of inputs), (list of expected)], ...]

        num_layers = len(self.weights)
        for age in range(epochs):
            grad_w = [np.zeros(w.shape) for w in self.weights]
            grad_b = [np.zeros(b.shape) for b in self.biases]
            for inputLayer, expected in inputs:
                grad_b_i, grad_w_i = self.backprop(inputLayer, expected)
                grad_w = [w + iw for w, iw in zip(grad_w, grad_w_i)]
                grad_b = [b + bw for b, bw in zip(grad_b, grad_b_i)]


            for i in range(num_layers):
                self.weights[i] -= grad_w[i]  * (learning_rate/num_layers)
                self.biases[i] -= grad_b[i] * (learning_rate/num_layers)

        return (self.weights, self.biases)
    
    def mse(self, data):
    #same data as backprop

        sum = 0
        for part in data:
            outputs = self.forward_pass(part[0])

            partialSum = 0

            for i in range(len(outputs)):
                partialSum += (outputs[i] - part[1][i]) ** 2

            sum += partialSum / len(outputs)
      
        return sum / len(data)
    
    def debug_descent(self, data, learning_rate, epochs, debug_rate):
        #debug rate is how often to send debug output
        #ie. 100 means once per 100 epochs
        
        # data is in the form:
        # [np.array([input1, input2, ...]), np.array([output1, output2, ...]), np.array(...)]

        data = [[np.atleast_2d(d[0]).transpose(), np.atleast_2d(d[1]).transpose()] for d in data]

        num_descent = math.ceil(epochs / debug_rate)
        prev_mse = None
        for i in range(num_descent):

            diff_string = ""
            mse = self.mse(data)
            if prev_mse != None:
                diff = mse - prev_mse
                colorCode = 32 if 0 > diff else 31
                symbol = '↓ ' if 0 > diff else '↑ '
                diff_string = " : " + '\u001b[' + str(colorCode) + 'm' + symbol + str(diff) + '\u001b[0m'

            print("MSE at epoch " + str(i*debug_rate) + ": " + str(mse) + diff_string)
            prev_mse = mse
            self.gradient_descent(data, learning_rate, debug_rate)
   
    def backprop(self, inputs, expected):
        # input is an np array of values for input neuron
        # expected is an np array of expected for output neurons

        zActivations, activations = self.full_forward_pass(inputs)       

        costs = 2*(activations[-1] - expected) * self.derivative(zActivations[-1])
        # if we don't start by doing the output layer manually,
        # making the loop work will be a pain bc one part only
        # happens for output layer

        gradient_w = [np.matmul(costs, activations[-2].transpose())] # weights
        gradient_b = [costs] # biases


        for idx in range(len(self.biases)-2, -1, -1):
            costs = np.matmul(self.weights[idx+1].transpose(), costs) * self.derivative(zActivations[idx])

            gradient_b.append(costs)

            gradient_w.append(np.matmul(costs, activations[idx].transpose()))

        gradient_w.reverse()
        gradient_b.reverse()
        return (gradient_b, gradient_w)