import tensorflow as tf, numpy as np

# most of this file is pretty much stolen from 
#https://gist.github.com/jakelevi1996/5c532463d59016f42d2bbdcfedd3372a

def initalize_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    np.savez("mnist", x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)
    return x_train, y_train


def get_data(filename="mnist.npz"):
    mnist_file = np.load(filename)
    x_train = mnist_file["x_train"]
    y_train = mnist_file["y_train"]
    # x_test = mnist_file["x_test"]
    # y_test = mnist_file["y_test"]
    return x_train, y_train



def training_data():
    data = []
    
    inputData, output = get_data()

    for i in range(len(inputData)):

        expected = np.zeros(10)
        expected[output[i]] = 1

        data.append([inputData[i].flatten(), expected])
    
    return data