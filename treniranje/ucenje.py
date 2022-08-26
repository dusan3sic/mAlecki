import numpy as np
import dataset
import koeficijenti

#load
X, y, tX, ty = dataset.load_data("./data.csv")

#process
def process(X):
    n = len(X) // 10
    for i, x in enumerate(X):
        if(i % n == 0): print("Proces: ", i)
        X[i] = koeficijenti.mfcc(x.astype("float64"))
        if(len(X[i]) != 13): print(i)
X = process(X)

# print(X)