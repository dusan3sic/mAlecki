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
    return x, y

def normalize(x: np.ndarray) -> np.ndarray:
    return (x - x.min()) / (x.max() - x.min())