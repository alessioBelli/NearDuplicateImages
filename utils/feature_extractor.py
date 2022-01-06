from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import numpy as np


class FeatureExtractor:
    def __init__(self):                              #costruttore per la classe
        base_model = ResNet50(weights='imagenet')    #il modello base con pesi pre-trained di ImageNet 
        self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output) #non mi interessa il modello completo ma utilizzo solo il layer avg_pool
    
    #estraggo deep feature dall'immagine in ingresso
    def extract(self, img):          
        #Args:
        #  img: from PIL.Image.open(path)
        #Returns:
        #  feature (np.ndarray): deep feature with the shape=(2048, )
        
        img = img.resize((224, 224))  # ResNet in ingresso deve avere 224x224
        img = img.convert('RGB')      # Mi assicuro che sia in RGB (ResNet opera sui 3 canali)
        x = image.img_to_array(img)   # rendo l'immagine un np.array. Dimensioni: Height x Width x Channel. dtype=float32
        x = np.expand_dims(x, axis=0)  # (H, W, C)->(1, H, W, C), il primo elemento sarà il numero dell'immagine
        x = preprocess_input(x)        # pre-processing (Subtracting avg values for each pixel)
        feature = self.model.predict(x)[0]    #considero solo la prima dimensione, ossia passo da (1, 2048) -> (2048, )
        return feature / np.linalg.norm(feature)  # Normalizzo il risultato
