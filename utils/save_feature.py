from utils.feature_extractor import FeatureExtractor
from PIL import Image
import numpy as np
import os
import sys

#Funzione per la creazione delle feature per ogni immagine del dataset
def creazioneFeature():
    fe = FeatureExtractor()
   
    #Controllo presenza cartella ./features, creazione se non ci dovesse essere
    if os.path.isdir("./features") == True:
        print("Directory già presente")
    else:
        os.mkdir("./features")

    image_names = os.listdir("gallery/")

    cont = 0
    
    #Estrazione feature per ogni immagine e salvataggio nella cartella ./feature
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
        feature = fe.extract(img=Image.open(f"gallery/{nomeFile}.jpg"))             #richiamo il metodo extract della classe esterna featureExtractor
        feature_path = f"features/{nomeFile}.npy"
        np.save(feature_path, feature)                                              #salvo nel percorso specificato l'array Numpy (file .npy)
