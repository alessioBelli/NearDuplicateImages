import cv2
import os
import numpy as np
from utils import descriptor
from sklearn.metrics.pairwise import cosine_similarity


# take second element for sort
def takeSecond(elem):
    return elem[1]


def istogrammi(input1, input2):
    fsRead= cv2.FileStorage ("inputHistograms/histograms.yml", cv2.FileStorage_READ )   #funzione per leggere dati dal file specificato
    compare=0 
    countc=0    #counter colonna
    countr=0    #counter riga
    nomeFile1 = os.path.splitext(input1)[0]
    nomeFile2 = os.path.splitext(input2)[0]
    for r in range(0,2):                                                                    #cicla per le due righe in cui ho suddiviso la finestra
        for c in range(0,2):                                                                #cicla per le due colonne in cui ho suddiviso la finestra
            hist_file1=fsRead.getNode(f'histogram_{r}_{c}_{nomeFile1}').mat()
            hist_file2=fsRead.getNode(f'histogram_{r}_{c}_{nomeFile2}').mat()					    
            compare+=cv2.compareHist(hist_file1,hist_file2, cv2.HISTCMP_CORREL)				#compara l'istogramma della finestra di query con la finestra del file dataset e la somma in count
            countc+= 1
        countr+=1
        countc=0

    media=compare/4																	#calcola la media 
    return media

def descriptor(input1, input2):
    
    nomeFile1 = os.path.splitext(input1)[0]
    nomeFile2 = os.path.splitext(input2)[0]
    desc1 = np.load(f"inputDescriptors/{nomeFile1}.npy")
    desc2 = np.load(f"inputDescriptors/{nomeFile2}.npy")
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    if(type(desc1) == type(None) or type(desc2) == type(None)):
        return 0
    else:   
        matches = bf.match(desc1,desc2)
        similar_regions = [i for i in matches if i.distance <50]
        if len(matches)==0:
            return 0
        return (len(similar_regions) / len(matches))

def features(input1, input2):
    nomeFile1 = os.path.splitext(input1)[0]
    feature1 = np.load(f"inputFeatures/{nomeFile1}.npy")
    
    nomeFile2 = os.path.splitext(input2)[0]
    feature2 = np.load(f"inputFeatures/{nomeFile2}.npy")
    cos_sim = cosine_similarity(feature1.reshape(1,-1), feature2.reshape(1,-1))
    return cos_sim


#comparazione tra due immagini
def compara(input1 ,input2):
    result_histograms = istogrammi(input1, input2)  #comparazione immagini secondo istogrammi

    result_orb = descriptor(input1, input2) #comparazione immagini secondo orb

    result_features = features(input1, input2)  #comparazione immagini secondo features                  

    result_similarity = ((result_orb*50 + result_features[0][0]*30 + result_histograms*20)) #risultato similarità 
    return result_similarity
    
