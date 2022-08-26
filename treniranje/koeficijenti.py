import numpy as np
import librosa.feature

n_mfcc = 13

def mfcc(niz):
    res = librosa.feature.mfcc(y = niz, sr = 48000, n_mels=14, n_fft=len(niz), hop_length=len(niz), n_mfcc=n_mfcc)
    return res