import numpy as np
import pandas as pd
import wave
import matplotlib.pyplot as plt
import os

def cut_clean(signal):
    brojac = 0
    while(True):
        if(brojac == len(signal) + 1): return []
        if(signal[- brojac - 1] == 0): brojac += 1
        else: return signal[:-brojac]

def separate(signal, n):
    x = len(signal) // 200
    signal = signal[:len(signal) - len(signal) % x]
    signal = signal.reshape(len(signal) // x, x)

    if(len(signal) > 200): signal = signal[:200-len(signal)]

    return signal

def signal(fileName, a):
    path = "./waves/" + fileName
    if(not path.endswith(".wav")): path += ".wav"
    raw = wave.open(path)

    signal = raw.readframes(-1)
    signal = np.frombuffer(signal, dtype ="int16")

    signal = cut_clean(signal)
    signal = separate(signal, 200)

    if(a == 0): indexes = [i for i, x in enumerate(signal)]
    else: indexes = [200 for i in signal]

    pd.DataFrame(signal, index=indexes).to_csv("./data", header=None, mode="a")

def ret_speed():
    return np.array(
        list(
            map(
                int, 
                np.linspace(100, 200, 25)
            )
        )
    )

def clearFile(name):
    pd.DataFrame([]).to_csv(name, header=None)

def empty_folder(folder):
    out = -1
    out = os.system('find . -name "' + folder + '" -d >/dev/null 2>/dev/null')
    if(out != -1): os.system("rm -rf " + folder)

    os.system("mkdir waves")

def make(name, data):
    #wav files making
    empty_folder("waves")
    speeds = ret_speed()
    for broj, i in enumerate(speeds):
        os.system("espeak " + name + " -s " + str(i) + " -w ./waves/" + name + str(broj + 1) + ".wav")
    
    #wav to csv
    clearFile(data)
    for i, _ in enumerate(speeds):
        signal(name + str(i + 1), 0)

    #done
    print("Done make!")

def make_poop(djubre):
    for i in djubre:
        os.system("espeak " + i + " -w ./waves/" + i  + ".wav")
        signal(i, 1)

    print("Done poop!")