# Import the libraries
import numpy as np
from utils.feature_extractor import FeatureExtractor 
from PIL import Image
import os
import re
from sklearn.metrics.pairwise import cosine_similarity

#funzione per ordinamento ascendente nomi file
def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

def compara(input):
    fe=FeatureExtractor()
    image_names = os.listdir("gallery/")
    
    #estrazione feature immagine di input
    imgInput = Image.open("images/"+input)                          
    featureInput = fe.extract(imgInput)             
    
    #confronto delle feature dell'immagine di input con tutte le feature delle immagini del dataset (feature salvate nella cartella features in formato npy)
    result = []
    for filename in image_names:
        nomeFile = os.path.splitext(filename)[0]
        result.append((nomeFile, cosine_similarity(featureInput.reshape(1,-1), np.load(f"features/{nomeFile}.npy").reshape(1,-1))))  #creazione array che contiene per ogni immagine del dataset la similarità con l'immagine in input
                        
    return result
