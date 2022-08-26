import numpy as np
import pandas as pd
import librosa.feature

n_mfcc = 13

def mfcc(niz):
    res = librosa.feature.mfcc(y = niz, sr = 48000, n_mels=30, n_fft=len(niz), hop_length=len(niz), n_mfcc=n_mfcc)
    return res

def process(X):
    n = len(X) // 10
    for i, x in enumerate(X):
        if(i % n == 0): print("Proces: ", i)
        X[i] = mfcc(x.astype("float64")).transpose()[0]

    return X

def saveProcess(X, tX):
    X = process(X)
    tX = process(tX)

    pd.DataFrame(tX).to_csv("tX.csv", header=None, mode="w")
    pd.DataFrame(X).to_csv("X.csv", header=None, mode="w")

