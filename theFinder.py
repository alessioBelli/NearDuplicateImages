import os
from flask import Flask, request, render_template, send_from_directory
from flask import Flask, session, request, redirect, url_for
from utils import saveHisto
from utils import save_feature
from utils import comparazione
from utils import descriptor
from utils import pairwise
from utils import pairwise
import string
import random
import shutil
import json


__author__ = 'AJA'

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

#Frammento di codice usato per gestire gli errori del Server (eccezione 404 page not found)
@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

#Visualizzazione index della web app
@app.route("/")
def index():
    image_names = os.listdir('./gallery')
    return render_template("index.html", numeroImage = len(image_names))

#Visualizzazione della pagine di upload
@app.route("/upload", methods=['GET'])
def upload_get():
    return render_template("upload.html")

@app.route("/groups")
def groups():
    groups = pairwise.createMatchMatrix()
    print(groups)
    return render_template("groups.html", dati=groups)

@app.route("/uploadFolder", methods=["POST"])
def uploadFolderPost():
    target = "./inputFolder"
    if not os.path.isdir(target):
        os.mkdir(target)
    else: 
        shutil.rmtree('./inputFolder', ignore_errors=True)
        os.mkdir(target)

    i = 0
    for upload in request.files.getlist("file"):
        #filename = upload.filename
        nameTemp = str(i)+".jpg"
        destination = "/".join([target, nameTemp])
        #Salvataggio immagini nella cartella ./images
        upload.save(destination)
        i = i+1
    
    if not os.path.isdir("./inputHistograms"):
        os.mkdir("./inputHistograms")
    else: 
        shutil.rmtree('./inputHistograms', ignore_errors=True)
        os.mkdir("./inputHistograms")
    
    target = "./inputDescriptors"
    if not os.path.isdir(target):
        os.mkdir(target)
    else: 
        shutil.rmtree(target, ignore_errors=True)
        os.mkdir(target)

    target = "./inputFeatures"
    if not os.path.isdir(target):
        os.mkdir(target)
    else: 
        shutil.rmtree(target, ignore_errors=True)
        os.mkdir(target)

    pairwise.createFiles()

    return redirect(url_for("groups"))

#Metodo richiamato quando si fa l'upload dell'immagine
@app.route("/upload", methods=["POST"])
def upload():

    #Controllo sessione dell'utente
    if not session.get('user') is None:
        print("Sessione già creata per l'utente")
    else:
        print("Nuovo Utente, creazione sessione")
        session["user"] = id_generator(10)
    print("Codice sessione corrente: "+session.get("user"))

    
    target = os.path.join(APP_ROOT, 'images/')
    #Creazione cartella ./images se non presente
    if not os.path.isdir(target):
        os.mkdir(target)

    #Salvataggio immagine caricata dall'utente nella cartella ./images
    for upload in request.files.getlist("file"):
        listImmagini = os.listdir("./images")
        print("Il file caricato è {}".format(upload.filename))
        filename = upload.filename
        estensione = filename[-3:]  #Prelievo estensione immagine utente
        filename = session.get("user") +"."+ estensione #Rinominazione immagine caricata dall'utente con il nome della sessione corrente (per utilizzo multi-utente contemporaneamente)
        #Sovrascrivere l'ultima immagine caricata da un utente (se presente)
        if filename in listImmagini:
            os.remove("./images/"+filename)
        session["quantity"] = request.form.get('quantity') #Numero di immagini simili da visualizzare (estratto dal form della pagina upload.html)
        session["lastImage"] = filename 
        destination = "/".join([target, filename])
        #Salvataggio immagine nella cartella ./images
        upload.save(destination)
        
    #Reindirizzamento alla pagina dei risultati
    return redirect(url_for("results"))

#Funzione per agire sulla cache
@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r

#Visualizzazione gallery.html 
@app.route('/gallery')
def get_gallery():
    image_names = os.listdir('./gallery')
    random.shuffle(image_names)
    return render_template("gallery.html", image_names=image_names)

@app.route('/duplicate')
def uploadFolder():
    return render_template("duplicate.html")

#Funzione che invia dal server al client le immagini della gallery
@app.route('/gallery/<filename>')
def send_image(filename):
    return send_from_directory("gallery", filename)

@app.route('/groups/<filename>')
def send_duplicates(filename):
    return send_from_directory("inputFolder", filename)

#Funzione chiamata una volta caricata l'immagine per estrarre le immagini simili dal dataset
def elaborazioneImmagini():
    #filenames è una stringa in formato json contenente le informazioni sulle immagini simili estratte della comparazione (contenente nome e percentuale per ogni immagine)
    filenames = comparazione.compara(session.get("lastImage"),session.get("quantity"))

    #Conversione da json a oggetto python
    dati = json.loads(filenames)

    #Approssimazione della percentuale ad una cifra decimale
    for elem in dati:
        elem["percentage"] = '%.1f'%(float(elem["percentage"]))
        elem["percentage_histo"] = '%.1f'%(float(elem["percentage_histo"]))
        elem["percentage_features"] = '%.1f'%(float(elem["percentage_features"]))
        elem["percentage_orb"] = '%.1f'%(float(elem["percentage_orb"]))
    return dati

#Visualizzazione della pagina con le immagini simili dopo la comparazione
@app.route('/results')
def results():
    #Controllo se sessione dell'utente già creata
    if session.get("user") == None:
        return render_template('404.html'), 404 #Ritorna una pagina personalizzata di errore

    dati = elaborazioneImmagini()

    #Invio dal server al client immagine di upload dell'utente e immagini simili
    return render_template("results.html",input_image=session.get("lastImage"), dati=dati)

#Funzione che invia dal server al client le immagini simili estratte
@app.route('/results/<filename>')
def send_image_results(filename):
    
    #Controllo per capire da che cartella prelevare le immagini richieste
    if filename in session.get("lastImage"):
        source = "images"
    else:
        source = "gallery"
    return send_from_directory(source, filename)

#Funzione per generare id casuale utente
def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

#Funzioni richiamate al momento della creazione del Server
if __name__ == "__main__":
    saveHisto.saveHisto()
     #solo se non esiste la cartella feature o se il numero di file in gallery è diverso dal numero di file in features, chiamo la funzione per il salvataggio di feature
    if os.path.isdir("./features") == False or len(os.listdir("./gallery"))!=len(os.listdir("./features")):
        save_feature.creazioneFeature()
    print("__CREAZIONE DESCRIPTOR__")
    descriptor.creaDescriptor()
    app.secret_key = 'super secret key'
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
    app.run(port=4555, debug=True, use_reloader=False)
