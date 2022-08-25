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
    return X, Y, tX, tY

def normalize(x: np.ndarray) -> np.ndarray:
    return (x - x.min()) / (x.max() - x.min())