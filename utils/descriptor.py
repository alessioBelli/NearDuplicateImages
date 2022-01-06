import numpy as np
import cv2
import os


#funzione che crea la cartella ./descriptor se non già presente
#e salva sotto forma di file npy i descriptor relativi alle immagini della galleria
def creaDescriptor():
    image_names = os.listdir("gallery/")
    if os.path.isdir("./descriptor") == True:
        print("Directory già presente")
    else:
        os.mkdir("./descriptor")
        
    for filename in image_names:
        nomeFile = os.path.splitext(filename)[0]
        image=cv2.imread(f"gallery/{filename}")   
        orb = cv2.ORB_create()
        kp_a, desc_a = orb.detectAndCompute(image,None)
        feature_path = f"descriptor/{nomeFile}.npy"
        np.save(feature_path, desc_a) 

class Descriptor:
    def __init__(self, filename, descriptor):
        self.filename = filename
        self.descriptor = descriptor

class Similarities:
    def __init__(self, filename, score):
        self.filename = filename
        self.score = score

def comparaDescriptor(original):
    image_names = os.listdir("gallery/")
    descriptor=[]
    for filename in image_names:
        nomeFile = os.path.splitext(filename)[0]
        descriptor.append(Descriptor(nomeFile,np.load(f"descriptor/{nomeFile}.npy")))       #lettura descriptor da file e salvataggio nell'array descriptor[]
        
    descriptor=np.array(descriptor,dtype=object) 
    orb = cv2.ORB_create()
    kp_a, desc_a = orb.detectAndCompute(cv2.imread("images/"+original,0),None)     #estrazione descriptor immagine di input
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    similaryties = []
    for a in descriptor:
        if(type(a) == type(None)):
            similaryties.append(Similarities(a.filename,0))
        else:   
            matches = bf.match(desc_a,a.descriptor)                                #comparazione descriptor immagine di input con descriptor immagini della galleria
            similar_regions = [i for i in matches if i.distance <50]
            if len(matches)==0:
                return 0
            similaryties.append(Similarities(a.filename,len(similar_regions) / len(matches)))
        
    return similaryties              #similaryties rappresenta una lista di oggetti di tipo Similarities