import csv
import numpy as np

def load_data(file):
    x = []
    y = []
    with open(file, 'r') as file:
        csvreader = csv.reader(file)
        for row in csvreader:
            row = list(map(int, row))
            y.append(row[0])
            x.append(row[1:])
    X = x[:10000]
    Y = y[:10000]
    tX = x[10000:]
    tY = y[10000:]
    return to_numpy(X), vectorize(Y), to_numpy(tX), vectorize(tY)

def load_processed(fileX, filetX):
    X = np.empty([10000, 13])
    tX = np.empty([1200, 13])
    with open(fileX, 'r') as file:
        csvreader = csv.reader(file)
        for row in csvreader: X[int(row[0])] = row[1:]
    
    with open(filetX, 'r') as file:
        csvreader = csv.reader(file)
        for row in csvreader: tX[int(row[0])] = row[1:]

    return X, tX
    

def normalize(x: np.ndarray) -> np.ndarray:
    if(x.min() == 0 and x.max() == 0): return x
    return (x - x.min()) / (x.max() - x.min())

def vectorize(y):
    vector_y = np.zeros((len(y), 201))
    for i in range (0, len(y)):
        vector_y[i, y[i]] = 1
    return vector_y

def to_numpy(niz):
    return [np.array(x) for x in niz]

