from network import *
from functions import *
from mnist import *
from cost_functions import *

def xor(a, b):
    return [np.array([a, b]), np.array([int(a!=b), 0])]

def andGate(a, b):
    return [np.array([a, b]), np.array([int(a and b)])]

def quadratic(a):
    return [np.array([a]), np.array([a**2])]

f = xor


def checkFunction(model, input, expected):
    print("Expected: " + str(expected) + " --- " + str(model.forward_pass(np.array(input))))



def get_best_network():
    net = Network([784, 15, 15, 15, 10], sigmoid, derivSigmoid, CrossEntropyLoss())
    net.load("mnist_network_two_prime")
    return net


data = training_data()

# net = Network([784, 15, 15, 10], sigmoid, derivSigmoid, CrossEntropyLoss())

# net.debug_descent(data, 0.0008, 50, 1, True)


# net.save("mnist_cel")

# data = [[np.atleast_2d(d[0]).transpose(), np.atleast_2d(d[1]).transpose()] for d in data]
if(__name__ == "__main__"):
    net = Network([784, 15, 15, 15, 10], sigmoid, derivSigmoid, CrossEntropyLoss())
    # net.load("mnist_network_two_prime")
    # net.debug_descent(data, 0.01, 50, 1, 100, True)
    # net.save("mnist_network_two_alpha")
    # print(data[10][1])

    net.debug_descent(data, 0.005, 50, 1, 100, True)
    net.save("mnist_test_batch")

# print(net.forward_pass(data[10][0]))