import math
import random

def ReLU(x): return (max(0, x))
def derivReLU(x): 
  return (1 if x > 0 else 0)

def leakyReLU(x): return (max(0.01*x, x))
def derivLeakyReLU(x):
  return (1 if x > 0 else 0.01)


def sigmoid(x): 
  # print(x)
  try:
    return (1/(1+(math.e**(-x))))
  except:
    print(x)
    raise Exception
def derivSigmoid(x):
  sig = sigmoid(x)
  return sig * (1-sig)



def gen_data(function, amount, negative=True):
  # function should take in one input and return one output

  start = -amount//2 if negative else 0
  end = -start if negative else amount

  data = []

  for i in range(start, end):
    data.append([[i], [function(i)]])
  random.shuffle(data)
  return data


def gen_range(function, min, max, amount):
  # same as gen_data but from min-max, doing floats

  data = []
  step = (max-min)/amount
  for i in range(amount):
    current = i*step + min
    data.append([[current], [function(current)]])
  return data