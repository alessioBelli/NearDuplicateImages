import cv2
import os
from os import path
import math

#Funzione per creare 4 istogrammi per ogni immagine del dataset
def saveHisto():
    
    fsWrite = cv2.FileStorage("utils/histograms.yml", cv2.FileStorage_WRITE )		    #funzione per creare il file 'histograms.yml' dove vengono salvati i dati

    #Controllo presenza file non voluto e eliminazione nel caso in cui sia presente
    if path.exists("gallery/.DS_Store"):
        os.remove("gallery/.DS_Store")
    
    image_names = os.listdir("gallery/")

    #Per ogni immagine nel nostro dataset, calcolo dei 4 istogrammi e salvataggio sul file histograms.yml
    
    for filename in image_names:
        nomeFile = os.path.splitext(filename)[0]					#separa il nome immagine dal formato
        image=cv2.imread(f"gallery/{filename}")						#legge l'immagine dalla galleria
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

    print("Salvataggio degli istogrammi eseguito correttamente")
    fsWrite.release()

