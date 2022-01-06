import math
import cv2
import os
from utils import query
from utils import descriptor


# take second element for sort
def takeSecond(elem):
    return elem[1]


def istogrammi(input):
    fsRead= cv2.FileStorage ("utils/histograms.yml", cv2.FileStorage_READ )			#funzione per leggere dati dal file specificato
    
    #calcolo istogramma immagine input
    image=cv2.imread(f"images/"+input)												#prende in input l'immagine
    image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)									#effettua la conversione dei colori in RGB
    
    hists={}                                                                        #creazione dizionario
    countc=0                                                                        #counter per numero colonna
    countr=0                                                                        #counter per numero riga
    windowsize_r = math.ceil(image.shape[0]/2)									    #crea una finestra per le righe di dimensione pari a metà della larghezza dell'immagine e la approssima per eccesso
    windowsize_c = math.ceil(image.shape[1]/2) 										#crea una finestra per le colonne di dimensione pari a metà dell'altezza dell'immagine e la approssima per eccesso
 
    for r in range(0,image.shape[0] , windowsize_r):											#cicla per spostare la finestra lungo tutta la larghezza dell'immagine
        for c in range(0,image.shape[1], windowsize_c):											#cicla per spostare la finestra lungo tutta l'altezza dell'immagine
            window = image[r:r+windowsize_r,c:c+windowsize_c]									#definizione della finestra 
            hist = cv2.calcHist([window],[0, 1, 2], None,[8, 8, 8],[0, 256, 0, 256, 0, 256]) 	#calcola l'istogramma associato alla finestra
            hists[f'histoquery_{countr}_{countc}'] = cv2.normalize(hist, hist).flatten()		#normalizza l'istogramma e lo salva nel dizionario 
            countc+= 1                                                                          
        countr+=1
        countc=0

    result_histograms = []
    image_names = os.listdir("gallery/")														#salva i nomi delle immagini
    for filename in image_names:
        nomeFile = os.path.splitext(filename)[0]												#separa il nome dell'immagine dal formato
        compare=0 
        for r in range(0,2):                                                                    #cicla per le due righe in cui ho suddiviso la finestra
            for c in range(0,2):                                                                #cicla per le due colonne in cui ho suddiviso la finestra
                hist_file=fsRead.getNode(f'histogram_{r}_{c}_{nomeFile}').mat()					#legge il file contenente i dati sull'istogramma di quella specifica finestra
                compare+=cv2.compareHist(hists[f'histoquery_{r}_{c}'],hist_file, cv2.HISTCMP_CORREL)				#compara l'istogramma della finestra di query con la finestra del file dataset e la somma in compare
                countc+= 1
            countr+=1
            countc=0
        
        media=compare/4																			#calcola la media 
        result_histograms.append((nomeFile, media))												#salva la media relativa nel dizionario 
    return result_histograms

        
#Funzione che ritorna una stringa json contente le _num_ immagini simili a quella di input con relativa percentuale
def compara(input,num):
    result_histograms = istogrammi(input)                       #richiamo metodo istogrammi() definito sopra dando in input l'immagine di query
    print("Fine comparazione istogrammi")

    result_orb = descriptor.comparaDescriptor(input)            #richiamo metodo comparaDescriptor() definito nel file descriptor.py che ritorna una lista di oggetti del tipo (filename, score)
    print("Fine comparazione descriptor")
    
    result_features = query.compara(input)                      #richiamo metodo compara() definito nel file query.py dando in input l'immagine di query
    print("Fine comparazione tramite feature")

    list_image = os.listdir("./gallery")
    n = len(list_image)                                         #numero immagini presenti nella cartella gallery
    
    res = []
    for i in range(len(list_image)):
        media = ((result_histograms[i][1])*0.2*100 + (result_features[i][1])*0.4*100 + (result_orb[i].score)*0.4*100)                          #20% peso istogrammi  40% features  40% orb
        res.append((result_features[i][0], media, result_histograms[i][1]*100, (result_features[i][1])*100, (result_orb[i].score)*100))        #appendo nome file, media pesata, percentuale istogrammi, percentuale feature e percentuale orb   
    
    #Riordinamento discendente (dal maggiore al minore) del vettore appena creato sulla base del secondo parametro (media pesata delle percentuali)       
    res.sort(key=takeSecond, reverse=True)

    #creazione JSON
    #-----------------------------------------------------------------------------
    count2=0
    filenames = "["
    #Creazione oggetto json contenente le num immagini simili da visualizzare
    for elem in res:
        count2+=1 
        if(count2 == int(num)):      #se è l'ultimo elemento (ossia count2 ha raggiunto il num di immagini che si vogliono visualizzare) non metto la virgola alla fine
            
            #creo oggetto in formato JSON contenente nomefile, percentuale pesata, percentuale isto, percentuale features
            filenames = filenames + '{ "name": "' + f"{elem[0]}.jpg" + '", "percentage": "' + ("%.3f" % (elem[1])) + '", "percentage_histo": "' + ("%.3f" % (elem[2])) + '", "percentage_features": "' + ("%.3f" % (elem[3])) + '", "percentage_orb": "' + ("%.3f" % (elem[4])) + '"}'
            break                    #esco dal for perchè hp raggiunto il numero di dati da passare in JSON
       
        else:                        #aggiungo virgola finchè ci sono altre stringhe da scrivere                    
            filenames = filenames + '{ "name": "' + f"{elem[0]}.jpg" + '", "percentage": "' + ("%.3f" % (elem[1])) + '", "percentage_histo": "' + ("%.3f" % (elem[2])) + '", "percentage_features": "' + ("%.3f" % (elem[3])) + '", "percentage_orb": "' + ("%.3f" % (elem[4])) + '"},'

    filenames = filenames + "]"      #chiudo la stringa JSON

    #restituisco la stringa di formato json appena creata
    return filenames        
    #-----------------------------------------------------------------------------
    
