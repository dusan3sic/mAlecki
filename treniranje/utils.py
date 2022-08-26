import numpy as np

def ReLU(data):
    return data * (data > 0)

def dReLU(data):
    return 1 * (data > 0)

def Softmax(data):
    mxs = np.max(data, axis = 1).reshape(-1, 1)
    exps = np.exp(data - mxs)

    sums = np.sum(exps, axis = 1).reshape(-1, 1)

    return exps / sums

def CELoss(y, yH): 
    return -np.sum(y * np.log(1e15 + yH) ) / y.shape[0]

def Randomize(X, y):
    index = np.arange(X.shape[0])
    np.random.shuffle(index)
    X = X[index]
    y = y[index]
    return X, y