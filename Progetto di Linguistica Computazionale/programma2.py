# -*- coding: utf-8 -*-
import sys
import codecs
import math
import nltk
from nltk import bigrams

lista_SostantiviPOS = ['NN', 'NNS', 'NNP', 'NNPS']
lista_VerbiPOS = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
lista_AggettiviPos = ['JJ', 'JJR', 'JJS']

def EstraiNE(POS):      #estrarre le entità nominate presenti nel testo
    nomi = []
    luoghi = []
    #creo dizionario
    analisi = nltk.ne_chunk(POS)
    for nodo in analisi:
        NE = ''
        if hasattr(nodo, 'label'):
            #distinguo i nomi propri di persona
            if nodo.label() == 'PERSON':
                for partNE in nodo.leaves():
                    NE = NE + ' ' + partNE[0]
                nomi.append(NE)
            #distingo i nomi propri di luogo
            if nodo.label() == 'GPE':
                for partNE in nodo.leaves():
                    NE = NE + ' ' + partNE[0]
                luoghi.append(NE)
    #ordino i nomi propri di persona per frequenza e prendo i primi 15
    distNomi = nltk.FreqDist(nomi)
    distNomiOrdinati = distNomi.most_common(15)
    #li stampo
    print("I 15 nomi propri di persona più frequenti:")
    for el in distNomiOrdinati:
        print(el[0], el[1])
    print()
    #ordino i nomi propri di luogo per frequenza e prendo i primi 15
    distLuoghi = nltk.FreqDist(luoghi)
    distLuoghiOrdinati = distLuoghi.most_common(15)
    #li stampo
    print("I 15 nomi propri di luogo più frequenti:")
    for el in distLuoghiOrdinati:
        print(el[0], el[1])

def Markov(LunghezzaCorpus, DistribuzioneDiFrequenzaToken, DistribuzioneDiFrequenzaBigrammi, bigrammiFrase, Vocabolario):
    #funzione per calcolare la probabilità di una frase usando una catena di markov di ordine 1 con add one smoothing
    token1 = bigrammiFrase[0][0]
    probabilita = (DistribuzioneDiFrequenzaToken[token1]*1.0/LunghezzaCorpus*1.0)
    for bigramma in bigrammiFrase:
        freqBigramma = (DistribuzioneDiFrequenzaBigrammi[bigramma])
        frequenzaA = DistribuzioneDiFrequenzaToken[bigramma[0]]
        probcondizionata = (freqBigramma*1.0 + 1)/(frequenzaA*1.0 + len(Vocabolario))
        probabilita=probabilita*probcondizionata
    return probabilita

def calcolostatistichebigrammi(bigrammi, tokens):
    #ordino i bigrammi per frequenza e prendo i primi 20
    distbigrammi = nltk.FreqDist(bigrammi)
    distbigrammiOrd = distbigrammi.most_common(20)
    print("20 bigrammi con relative probabilità:")
    print()
    #per ogni bigramma calcolo quante volte i tokens che compongono il bigramma compaiono nel testo
    for bigramma in distbigrammiOrd:
        token1 = bigramma[0][0]
        token2 = bigramma[0][1]
        freqtoken1 = tokens.count(token1)
        freqtoken2 = tokens.count(token2)
        freqBigramma = bigramma[1]
        #i tokens del bigramma devono avere frequenza maggiore di 3
        if (freqtoken1 > 3) and (freqtoken2 > 3):
            #calcolo probabilità condizionata
            probcondizionata = freqBigramma*1.0/freqtoken1*1.0
            probtoken1 = freqtoken1*1.0/len(tokens)*1.0
            probtoken2 = freqtoken2*1.0/len(tokens)*1.0
            #calcolo probabilita congiunta
            probCongiunta = probcondizionata*probtoken1
            print("Bigramma: (", token1, token2, ")")
            print("Frequenza Bigramma:", freqBigramma)
            print("Probabilità condizionata:", probcondizionata)
            print("Probabilità congiunta:", probCongiunta)
            #calcolo la local mutual information
            p = probCongiunta*1.0/(probtoken1*probtoken2)*1.0
            mi = math.log(p,2)
            print("LMI:", mi)
            print()


def EseguiStatistiche(POS, bigrammi_POS):
    #ordino i tokens per frequenza e prendo i 10 più frequenti
    distPOS = nltk.FreqDist(POS)
    distPOSordinate = distPOS.most_common(10)
    #estraggo le 10 part-of-speech più frequenti
    print('Le 10 Part-of-Speech più frequenti:')
    for el in distPOSordinate:
        partofspeech = el[0][1]
        freq = el[1]
        print('PoS:', partofspeech, '- Frequenza:', freq)
    #creo lista dei sostantivi
    sostantivi_POS = []
    for big in POS:
        if big[1] in lista_SostantiviPOS:
            sostantivi_POS.append(big)
    #prendo i 10 sostantivi più frequenti
    dist_sostantivi_POS = nltk.FreqDist(sostantivi_POS)
    dist_sostantivi_POS_ordinati = dist_sostantivi_POS.most_common(20)
    #creo lista dei verbi
    verbi_POS = []
    for big in POS:
        if big[1] in lista_VerbiPOS:
            verbi_POS.append(big)
    #prendo i 10 verbi più frequenti
    dist_verbi_POS = nltk.FreqDist(verbi_POS)
    dist_verbi_POS_ordinati = dist_verbi_POS.most_common(20)
    #stampo i sostantivi e i verbi assieme alla loro frequenza
    print()
    print('I 20 sostantivi più frequenti:')
    for el in dist_sostantivi_POS_ordinati:
        token = el[0][0]
        freq = el[1]
        print('Sostantivo:', token, 'Frequenza:', freq)
    print()
    print('I 20 verbi più frequenti:')
    for el in dist_verbi_POS_ordinati:
        token = el[0][0]
        freq = el[1]
        print('Verbo:', token, 'Frequenza:', freq)
    #creo lista dei bigrammi in cui il primo token è un sostantivo e il secondo token è un verbo
    bigrammi_sostantivoverbo = []
    for big in bigrammi_POS:
        pos1 = big[0][1]
        pos2 = big[1][1]
        if pos1 in lista_SostantiviPOS and pos2 in lista_VerbiPOS:
            bigrammi_sostantivoverbo.append(big)
    #prendo i bigrammi più frequenti
    dist_bigrammi_sostantivoverbo = nltk.FreqDist(bigrammi_sostantivoverbo)
    dist_bigrammi_sostantivoverbo_ord = dist_bigrammi_sostantivoverbo.most_common(20)
    print()
    #stampo i bigrammi sostantivo + verbo seguito dalla loro frequenza
    print('I 20 bigrammi composti da un Sostantivo seguito da un Verbo più frequenti:')
    for el in dist_bigrammi_sostantivoverbo_ord:
        token1 = el[0][0][0]
        token2 = el[0][1][0]
        freq = el[1]
        print('Bigramma:', token1, token2, 'con frequenza:', freq)
    #creo lista dei bigrammi in cui il primo token è un aggettivo e il secondo token è un sostantivo
    bigrammi_aggettivosostantivo = []
    for big in bigrammi_POS:
        pos1 = big[0][1]
        pos2 = big[1][1]
        if pos1 in lista_AggettiviPos and pos2 in lista_SostantiviPOS:
            bigrammi_aggettivosostantivo.append(big)
    #prendo i bigrammi più frequenti
    dist_bigrammi_aggettivosostantivo = nltk.FreqDist(bigrammi_aggettivosostantivo)
    dist_bigrammi_aggettivosostantivo_ord = dist_bigrammi_aggettivosostantivo.most_common(20)
    print()
    #stampo i bigrammi aggettivo + verbo seguito dalla loro frequenza
    print('I 20 bigrammi composti da un Aggettivo seguito da un sostantivo più frequenti:')
    for el in dist_bigrammi_aggettivosostantivo_ord:
        token1 = el[0][0][0]
        token2 = el[0][1][0]
        freq = el[1]
        print('Bigramma:', token1, token2, 'con frequenza:', freq)
    


def AnalisiTesto(frasi):        #dividere le frasi in tokens e in bigrammi
    tokensTOT = []
    tokensPOStot = []
    bigrammiPOS_TOT = []
    bigrammi_TOT = []
    for frase in frasi:
        tokens = nltk.word_tokenize(frase)
        tokensPOS = nltk.pos_tag(tokens)
        tokensPOStot+=tokensPOS
        bigrammi = bigrams(tokens)
        bigrammi_POS = bigrams(tokensPOS)
        bigrammiPOS_TOT+=bigrammi_POS
        bigrammi_TOT+=bigrammi
        tokensTOT+=tokens

    return tokensTOT, tokensPOStot, bigrammiPOS_TOT, bigrammi_TOT

def main(file1, file2):
    fileInput1 = codecs.open(file1, "r", "utf-8")
    fileInput2 = codecs.open(file2, "r", "utf-8")
    raw1 = fileInput1.read()
    raw2 = fileInput2.read()
    sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    frasi1 = sent_tokenizer.tokenize(raw1)
    frasi2 = sent_tokenizer.tokenize(raw2)

    tokens1, TestoPOS1, bigrammiPOS1, bigrammi1 = AnalisiTesto(frasi1)
    tokens2, TestoPOS2, bigrammiPOS2, bigrammi2 = AnalisiTesto(frasi2)
    Vocabolario1 = list(set(tokens1))
    Vocabolario2 = list(set(tokens2))
    LunghezzaTesto1 = len(tokens1)
    LunghezzaTesto2 = len(tokens2)
    DistribuzioneDiFrequenzaToken1=nltk.FreqDist(tokens1)
    DistribuzioneDiFrequenzaToken2=nltk.FreqDist(tokens2)
    DistribuzioneDiFrequenzaBigrammi1=nltk.FreqDist(bigrammi1)
    DistribuzioneDiFrequenzaBigrammi2=nltk.FreqDist(bigrammi2)
    print("File di testo:", file1)
    print()
    EseguiStatistiche(TestoPOS1, bigrammiPOS1)
    print()
    calcolostatistichebigrammi(bigrammi1, tokens1)
    print()
    print("Frasi lunghe da 8 a 15 tokens con probabilità più alta usando un modello markoviano di ordine 1:")
    print()
    for i in range(8, 16):
        #creo lista di frasi lunghe i
        frasiok = []
        for frase in frasi1:
            tokens = nltk.word_tokenize(frase)
            if len(tokens) == i:
                frasiok.append(frase)
        tokensok_tot = []
        bigrammiok_tot = []
        #tokenizzo e divido in bigrammi le frasi lunghe i
        for frase in frasiok:
            tokens = nltk.word_tokenize(frase)
            bigrammi = bigrams(tokens)
            tokensok_tot += tokens
            bigrammiok_tot += bigrammi
        prob1 = 0.0
        frase1 = []
        for frase in frasiok:
            tokensFrase = nltk.word_tokenize(frase)
            bigrammiFrase = list(bigrams(tokensFrase))
            #calcolo la probabilità della frase richiamando la funzione precedentemente creata
            probabilita = Markov(LunghezzaTesto1, DistribuzioneDiFrequenzaToken1, DistribuzioneDiFrequenzaBigrammi1, bigrammiFrase, Vocabolario1)
            #scelgo la frase con probabilità più alta
            if probabilita > prob1 :
                frase1 = frase
                prob1 = probabilita
        print(frase1, "è la frase lunga", i , "con probabiltà più alta e con probabilità:", prob1)
    print()
    EstraiNE(TestoPOS1)
    print()
    print("File di testo:", file2)
    print()
    EseguiStatistiche(TestoPOS2, bigrammiPOS2)
    print()
    calcolostatistichebigrammi(bigrammi2, tokens2)
    print()
    print("Frasi lunghe da 8 a 15 tokens con probabilità più alta usando un modello markoviano di ordine 1:")
    print()
    for i in range(8, 16):
        #creo lista di frasi lunghe i
        frasiok = []
        for frase in frasi2:
            tokens = nltk.word_tokenize(frase)
            if len(tokens) == i:
                frasiok.append(frase)
        tokensok_tot = []
        bigrammiok_tot = []
        #tokenizzo e divido in bigrammi le frasi lunghe i
        for frase in frasiok:
            tokens = nltk.word_tokenize(frase)
            bigrammi = bigrams(tokens)
            tokensok_tot += tokens
            bigrammiok_tot += bigrammi
        prob2 = 0.0
        frase2 = []
        for frase in frasiok:
            tokensFrase = nltk.word_tokenize(frase)
            bigrammiFrase = list(bigrams(tokensFrase))
            #calcolo la probabilità della frase richiamando la funzione precedentemente creata
            probabilita = Markov(LunghezzaTesto2, DistribuzioneDiFrequenzaToken2, DistribuzioneDiFrequenzaBigrammi2, bigrammiFrase, Vocabolario2)
            #scelgo la frase con probabilità più alta
            if probabilita > prob2 :
                frase2 = frase
                prob2 = probabilita
        print(frase2, "è la frase lunga", i , "con probabiltà più alta e con probabilità:", prob2)
    print()
    EstraiNE(TestoPOS2)
    
    
main(sys.argv[1], sys.argv[2])