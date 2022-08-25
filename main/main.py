import numpy as np
import dataset
import utils
from asm import *

end_state = 200
sensitivity = 3 / 100
razlika = int(end_state * sensitivity)
minSigurno = 0.65
state = 0
A = -1
B = []
pocetak = []
kraj = []
tempo = 0
brojac = 0

radi = 0
while(end_state - state >= razlika): #ponavljaj dok ne dodje do kraja... ili umre
    print(state, radi)
    if(radi == 0):    #algoritam ima dva stanja 1 i 0, 0 kad je na pocetku i 1 kada nastavlja sa radom
        state = 0
        tempo = 0
        brojac = 0

        pocetak = generisiProbable()  #random generacija procenata pocetka i kraja 
        kraj = generisiProbable()

        A = resenjePocetak(pocetak, 0)  #da li je ovaj segment na pocetku (-1 ako ne ili index pocetka)
        B = resenjeKraj(kraj, 0)        #na koje sve krajeve segmenata lici ovaj segment

        if(A != -1): radi = 1           #ako pocetak odgovoara nastavlajmo u sledecu fazu
        else:
            pocetak = generisiProbable()    
            for i in range(-(razlika // 2), razlika // 2):
                d = tempo + i
                if(d >= 0 and d < len(kraj)): kraj[d] *= 1.5
            maks = -1
            indeks = -1

            for i, b in enumerate(B): #prolazanje kroz svaki potencijalni zavrsetak prethodnog segmenta i gledanje da li pocetak novog odgovara nekom
                A = resenjePocetak(pocetak, b) 
                if(A != -1):
                    if(pocetak[A] + kraj[b] > maks):
                        indeks = A
                        maks = pocetak[A] + kraj[b]
      
            if(indeks != -1):                   #ako nadje bilo sta uzima najverovatniji
                kraj = generisiProbable()         
                B = resenjeKraj(kraj, indeks)     
                tempo = (brojac * tempo + indeks - state) // max(brojac, 1)     #racunanje prosecne brzine pricanja reci
                brojac += 1
                state = indeks        
            else:                             
                A = resenjePocetak(pocetak, 0)              #gledanje da li je ovo pocetak reci
                if(A == -1): radi = 0
                else:                                       
                    kraj = generisiProbable()
                    B = resenjeKraj(kraj, 0)
                    state = 0
                    tempo = 0
                    brojac = 0
  
  

activate()          #jedini nacin da se zavrsi petlja je da se dodje do kraja asm-a