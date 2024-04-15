# -*- coding: utf-8 -*-
import sys
import codecs
import nltk
from nltk import bigrams
ListaPunteggiatura = ['.', ',', ':', ';', '!', '?']
lista_SostantiviPOS = ['NN', 'NNS', 'NNP', 'NNPS']
lista_VerbiPOS = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
lista_AvverbiPOS = ['RB', 'RBR', 'RBS', 'WRB']
lista_AggettiviPos = ['JJ', 'JJR', 'JJS']

def PuntoSei(tokensPOS, numerotokens):      #Densità lessicale calcolata secondo questa formula (|Sostantivi|+|Verbi|+|Avverbi|+|Aggettivi|)/(TOT-( |.|+|,| ) )
    nsostantivi = 0.0                                   
    nverbi = 0.0
    navverbi = 0.0
    naggettivi = 0.0
    nparolemenopunt = numerotokens
    #conto i sostantivi, verbi, avverbi, aggettivi
    for el in tokensPOS:                                 
        if el[1] in lista_SostantiviPOS:
            nsostantivi+=1
        if el[1] in lista_VerbiPOS:
            nverbi+=1
        if el[1] in lista_AvverbiPOS:
            navverbi+=1
        if el[1] in lista_AggettiviPos:
            naggettivi+=1
    #escludo il punto e la virgola
    for el in tokensPOS:            
        if el[0] in ['.', ',']:
            nparolemenopunt-=1
    #applico la formula sopra
    denslessicale = (nsostantivi + nverbi + navverbi + naggettivi)/(nparolemenopunt*1.0)        

    return denslessicale



def PuntoCinque(tokensPOS, numerofrasi):        #media di sostantivi e verbi per frase
    sostantivi = []
    verbi = []
    numerosostantivi = 0.0
    numeroverbi = 0.0
    #conto i sostantivi e i verbi
    for el in tokensPOS:        
        if el[1] in lista_SostantiviPOS:
            sostantivi+=el
            numerosostantivi+=1
        if el[1] in lista_VerbiPOS:
            verbi+=el
            numeroverbi+=1
    #calcolo la media per ogni frase
    mediasostantivi = numerosostantivi * 1.0 / numerofrasi      
    mediaverbi = numeroverbi * 1.0 / numerofrasi

    return mediasostantivi, mediaverbi



def PuntoQuattro(tokens, lunghezza):        #Distribuzioni classi di frequenza 1,5,10 ad aumentare corpus
    #ciclo che aumenta di 500 ad ogni iterazione
    for i in range(0, len(tokens), 500):        
        listaToken500 = tokens[0:i+500]
        Vocabolario = list(set(listaToken500))
        hapax = []
        V5 = []
        V10 = []
        for tok in Vocabolario:
            conteggio = listaToken500.count(tok)
            #divido i token in base alla loro frequenza 1, 5 o 10
            if conteggio == 1:                      
                hapax.append(tok)
            if conteggio == 5:
                V5.append(tok)
            if conteggio == 10:
                V10.append(tok)
        print(0, "-", len(listaToken500))
        #calcolo la distribuzione per le tre classi
        distHapax = len(hapax)* 1.0 / len(listaToken500)        
        distV5 = len(V5)* 1.0 / len(listaToken500)
        distV10 = len(V10)* 1.0 / len(listaToken500)
        print("Distribuzione degli hapax:", distHapax)
        print("Distribuzione classe di frequenza 5", distV5)
        print("Distribuzione classe di frequenza 10", distV10)
        print()

def PuntoTre(tokens, lunghezza):        #Vocabolario e TTR dei primi 5000 token
    tokens5000 = tokens[0:4999]
    vocabolario5000 = list(set(tokens5000))
    grandezzavocabolario = len(vocabolario5000)
    #calcolo la type-token-ratio 
    ttr = grandezzavocabolario * 1.0 / len(tokens5000)      
    return grandezzavocabolario, ttr

def PuntoDue(lunghezzaTot, numerofrasi, tokens):        #Lunghezza media frasi e parole
    lunghezzamediafrasi = lunghezzaTot/numerofrasi
    numerocaratteri = 0.0
    listatokensenzapunt = []
    lunghezzamediaparole = 0.0
    lunghezzalistapunt = 0.0
    #per calcolare la lunghezza media delle parole, escludo la punteggiatura
    for tok in tokens:                                  
        if tok not in ListaPunteggiatura:
            listatokensenzapunt+=tokens
    for tok in listatokensenzapunt:
        numerocaratteri+=len(tok)
    lunghezzalistapunt+=len(listatokensenzapunt)
    #calcolo la media
    lunghezzamediaparole = numerocaratteri/lunghezzalistapunt
    return lunghezzamediafrasi, lunghezzamediaparole

def PuntoUno(frasi):        #Calcolare il numero di frasi e di token
    tokensTot = []
    numerofrasi = 0.0
    lunghezzaTot = 0.0
    tokensPOStot = []
    #Per ogni frase nel testo divido in tokens e per ogni token il POS
    for frase in frasi:                                    
        tokens = nltk.word_tokenize(frase)      
        tokensPOS = nltk.pos_tag(tokens)
        tokensTot+=tokens
        tokensPOStot+=tokensPOS
        numerofrasi+=1
        lunghezzaTot+=len(tokens)
    return tokensTot, numerofrasi, lunghezzaTot, tokensPOStot

def main(file1, file2):
    sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    fileInput1 = codecs.open(file1, "r", "utf-8")
    fileInput2 = codecs.open(file2, "r", "utf-8")
    raw1 = fileInput1.read()
    raw2 = fileInput2.read()
    frasi1 = sent_tokenizer.tokenize(raw1)
    frasi2 = sent_tokenizer.tokenize(raw2)

    print('File di testo:', file1)      #Testo 1
    tokens1, numerofrasi1, numerotokens1, tokensPOS1 = PuntoUno(frasi1)
    lunghezzamediafrasi1, lunghezzamediaparole1 = PuntoDue(numerotokens1, numerofrasi1, tokens1)
    grandezzavocabolario1, TTR1 = PuntoTre(tokens1, numerotokens1)
    mediasostantivi1, mediaverbi1 = PuntoCinque(tokensPOS1, numerofrasi1)
    denslessicale1 = PuntoSei(tokensPOS1, numerotokens1)

    print()

    print(file1, "contiene", int(numerofrasi1), "frasi e", int(numerotokens1), "tokens.")
    print()
    print("La lunghezza media delle frasi in termini di token del file:", file1, "è", lunghezzamediafrasi1)
    print()
    print("La lunghezza media delle parole in termini di caratteri del file:", file1, "è", lunghezzamediaparole1)
    print()
    print("Il vocabolario del file:", file1, "è grande:", grandezzavocabolario1, ". La Type-Token ratio è:", TTR1)
    print()
    print("La distribuzione delle classi di frequenza |V1|, |V5| e |V10| all'aumentare del corpus per porzioni incrementali di 500 token:")
    PuntoQuattro(tokens1, numerotokens1)
    print()
    print("Le frasi del file:", file1, "hanno in media:", mediasostantivi1, "sostantivi e ", mediaverbi1, "verbi.")
    print()
    print("La densità lessicale calcolata secondo il punto 6 è:", denslessicale1)

    print()

    print('File di testo:', file2)      #Testo 2
    tokens2, numerofrasi2, numerotokens2, tokensPOS2 = PuntoUno(frasi2)
    lunghezzamediafrasi2, lunghezzamediaparole2 = PuntoDue(numerotokens2, numerofrasi2, tokens2)
    grandezzavocabolario2, TTR2 = PuntoTre(tokens2, numerotokens2)
    mediasostantivi2, mediaverbi2 = PuntoCinque(tokensPOS2, numerofrasi2)
    denslessicale2 = PuntoSei(tokensPOS2, numerotokens2)

    print()

    print(file2, "contiene", int(numerofrasi2), "frasi e", int(numerotokens2), "tokens.")
    print()
    print("La lunghezza media delle frasi in termini di token del file:", file2, "è", lunghezzamediafrasi2)
    print()
    print("La lunghezza media delle parole in termini di caratteri del file:", file2, "è", lunghezzamediaparole2)
    print()
    print("Il vocabolario del file:", file2, "è grande:", grandezzavocabolario2, ". La Type-Token ratio è:", TTR2)
    print()
    print("la distribuzione delle classi di frequenza |V1|, |V5| e |V10| all'aumentare del corpus per porzioni incrementali di 500 token:")
    PuntoQuattro(tokens2, numerotokens2)
    print()
    print("Le frasi del file:", file2, "hanno in media:", mediasostantivi2, "sostantivi e ", mediaverbi2, "verbi.")
    print()
    print("La densità lessicale calcolata secondo il punto 6 è:", denslessicale2)
main(sys.argv[1], sys.argv[2])