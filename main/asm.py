import numpy as np

def activate():
  print("AKTIVIRAN SAM IDE GAS SMRT SVETU")

def resenjeKraj(niz, state):
    peaks = []
    zadnji = 0

    for i in range(state, len(niz) - 1):
      procenat = niz[i]
      if(procenat > minSigurno):
        if(procenat > zadnji and procenat > niz[i + 1]):
          peaks.append(i)
        zadnji = procenat
      else: zadnji = 0
    
    return peaks

def resenjePocetak(niz, state):
    
    #gledanje od nastavka
    maxi = 0
    res = -1
    for i in range(-2, razlika - 2):
      index = i + state
      if(index > 0 and index < len(niz) - 1):
        if(niz[index] > maxi and niz[index] > minSigurno):
          maxi = niz[index]
          res = index
        elif(niz[index] == maxi and abs(state - res) > abs(state - index)): res = index


    if(maxi > minSigurno): return res
    return -1

def generisiProbable():

  niz = np.zeros(200)
  niz = niz + 0.1

  elementi = []

  brojSigurnih = random.randint(1, 5)
  while(len(elementi) <= brojSigurnih):
    index = random.randint(0, len(niz) - 1)
    povecaj = random.random() * (1 - minSigurno) + minSigurno
    if(niz[index] < minSigurno): elementi.append([index, povecaj])
  
  window = signal.windows.hann(5)
  window *= (1 - 0.92)
  window += 0.92

  for i, e in enumerate(elementi):
    for j in range(5):
      povecaj = window[j] * e[1]
      niz[min(len(niz) - 1, e[0] + j)] = povecaj
  
  return niz