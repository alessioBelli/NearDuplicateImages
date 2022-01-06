import numpy as np
import os
import cv2
from utils.feature_extractor import FeatureExtractor
from utils import comparazionePairwise
from PIL import Image
import sys
import math
from os import path

#funzione di normalizzazione della matrice
#input: matrice con i valori di similarità, soglia
#output: matrice di 0 (no near-duplicate) e 1 (near-duplicate)
def normalizeMatrix(pairwiseMatch,threshold):
    matchMatrix = np.zeros((len(pairwiseMatch),len(pairwiseMatch)))
    for i in range (len(pairwiseMatch)):
        for j in range (i+1,len(pairwiseMatch)):
            if(pairwiseMatch[i][j] > threshold):    #se il valore è maggiore della soglia, diventa 1
                matchMatrix[i][j] = 1
            else:                                   #altrimenti diventa 0
                matchMatrix[i][j] = 0

    return matchMatrix

#funzione di ricerca dei gruppi di near-duplicate
#input: matrice di similarità
#output: lista di gruppi
def multimatch(matchMatrix,threshold):
    matchGroups = []
    normalizedMatchMatrix = normalizeMatrix(matchMatrix,threshold)                              #normalizza la matrice
    n = len(normalizedMatchMatrix)
    for i in range (n):
        for j in range (i+1,n):
            if(normalizedMatchMatrix[i][j]==1):                                                 #se trovo una coppia near-duplicate
                tempSet = {i,j}                                                                 #creo il set composto dalle sue coordinate
                foundFlag = -1                                                                  #flag che indica se una delle due coordinate è già presente in un altro gruppo
                for k in range (len(matchGroups)):
                    if (tempSet.intersection(matchGroups[k]) != set() and foundFlag == -1):     #se una delle coordinate è già presente in un altro gruppo faccio un merge tra il gruppo esistente e il temporaneo
                        foundFlag = k
                        matchGroups[k] = matchGroups[k].union(tempSet)
                    elif (tempSet.intersection(matchGroups[k]) != set() and foundFlag != -1):   #se una delle coordinate è già presente in un altro gruppo e ho già eseguito il merge faccio un altro merge tra i due gruppi già esistenti
                        matchGroups[foundFlag] = matchGroups[foundFlag].union(matchGroups[k])
                        matchGroups[k] = set()
                if(foundFlag==-1):                                                              #se le coordinate non erano già in altri gruppi il gruppo temporaneo viene aggiunto alla lista
                    matchGroups.append(tempSet)
    return matchGroups


def createMatchMatrix():
    list_img = os.listdir("./inputFolder")
    n = len(list_img)
    matchMatrix = np.zeros((n,n))
    for i in range (n):
        for j in range (i+1,n):
            matchMatrix[i][j] = round(comparazionePairwise.compara(list_img[i],list_img[j]),2)
    #return matchGroups
    groups = multimatch(matchMatrix,50)
    imageList = []
    for i in range(len(groups)):
        imageList.append(list(groups[i]))
        for j in range(len(imageList[i])):
            imageList[i][j] = list_img[imageList[i][j]]
    return imageList

#funzione che salva su file features, istrogrammi e descriptors delle immagini caricate dall'utente
def createFiles():
    #CREAZIONE FEATURE
    fe = FeatureExtractor()

    #Controllo presenza file non voluto e eliminazione nel caso in cui sia presente
    if path.exists("inputFolder/.DS_Store"):
        os.remove("inputFolder/.DS_Store")

    image_names = os.listdir("inputFolder/")
    cont = 0

    #salvataggio nella cartella /inputFeatures delle feature relative alle immagini caricate dall'utente in formato npy
    for filename in image_names:
        cont = cont+1
        #Visualizzazione numero immagini (a intervalli di 10) eleborate nella console
        if cont % 10 == 0:
            sys.stdout.write('\r' + "Le immagini elaborate sono attualmente : " + str(cont) + "/" + str(len(image_names)))
            sys.stdout.flush()
        if cont == len(image_names):
            sys.stdout.write('\r' + "Le immagini elaborate sono attualmente : " + str(cont) + "/" + str(len(image_names)))
            sys.stdout.flush()

        nomeFile = os.path.splitext(filename)[0]
        feature = fe.extract(img=Image.open(f"inputFolder/{nomeFile}.jpg"))             #richiamo il metodo extract della classe esterna featureExtractor per estrarre le feature
        feature_path = f"inputFeatures/{nomeFile}.npy"
        np.save(feature_path, feature)

    #CREAZIONE ISTOGRAMMI
    fsWrite = cv2.FileStorage("inputHistograms/histograms.yml", cv2.FileStorage_WRITE )		    #funzione per creare il file 'histograms.yml' dove vengono salvati i dati

    #Per ogni immagine caricata, calcolo dei 4 istogrammi e salvataggio sul file inputHistograms/histograms.yml
    for filename in image_names:
        nomeFile = os.path.splitext(filename)[0]					#separa il nome immagine dal formato
        image=cv2.imread(f"inputFolder/{filename}")					#legge l'immagine dalla cartella /inputFolder
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)				#converte i colori dell'immagine

        windowsize_r = math.ceil(image.shape[0]/2)					#crea una finestra per le righe di dimensione pari a metà della larghezza dell'immagine e la approssima per eccesso
        windowsize_c = math.ceil(image.shape[1]/2)					#crea una finestra per le colonne di dimensione pari a metà dell'altezza dell'immagine e la approssima all'intero per eccesso

        countc=0                                                    #counter per numero colonna
        countr=0
        for r in range(0,image.shape[0] , windowsize_r):                                                #cicla per spostare la finestra lungo tutta la larghezza dell'immagine
            for c in range(0,image.shape[1], windowsize_c):                                             #cicla per spostare la finestra lungo tutta l'altezza dell'immagine
                window = image[r:r+windowsize_r,c:c+windowsize_c]										#definizione della finestra
                hist = cv2.calcHist([window],[0, 1, 2], None,[8, 8, 8],[0, 256, 0, 256, 0, 256])		#calcola l'istogramma delle relativa finestra
                hist = cv2.normalize(hist, hist).flatten()												#normalizza l'istogramma
                fsWrite.write(f'histogram_{countr}_{countc}_{nomeFile}' ,hist)							#salva l'istogramma della finestra, il numero della riga e della colonna
                countc+= 1
            countr+=1
            countc=0

    print("\nSalvataggio degli istogrammi eseguito correttamente")
    fsWrite.release()

    #creazione descriptor e salvataggio all'interno della cartella inputDescriptors/
    image_names = os.listdir("inputFolder/")
    for filename in image_names:
        nomeFile = os.path.splitext(filename)[0]
        image=cv2.imread(f"inputFolder/{filename}")
        orb = cv2.ORB_create()
        kp_a, desc_a = orb.detectAndCompute(image,None)
        feature_path = f"inputDescriptors/{nomeFile}.npy"
        np.save(feature_path, desc_a)
