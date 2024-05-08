from network import *
from functions import *
from mnist import *

def xor(a, b):
    return [np.array([a, b]), np.array([int(a!=b), 0])]

def andGate(a, b):
    return [np.array([a, b]), np.array([int(a and b)])]

def quadratic(a):
    return [np.array([a]), np.array([a**2])]

f = xor


def checkFunction(model, input, expected):
    print("Expected: " + str(expected) + " --- " + str(model.forward_pass(np.array(input))))




data = training_data()

net = Network([784, 50, 10, 10], sigmoid, derivSigmoid)

net.debug_descent(data, 0.01, 100000, 100)