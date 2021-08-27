import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"   
import tensorflow as tf
import json
import cv2
import numpy as np

class Pipeline:

    def __init__(self, model_config_path, weights_path):
        self.model = self.initialise_model(model_config_path, weights_path)

    def initialise_model(self, model_config_path, weights_path):
        with open(model_config_path, 'r') as f:
            model_config = f.read()
        model_cfg = json.loads(model_config)
        model = tf.keras.models.model_from_json(model_config)

        model.load_weights(weights_path)

        return model

    def preprocessing_image(self, img_path):
        try:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224,224), cv2.INTER_AREA)
            img = tf.keras.applications.resnet.preprocess_input(img)
            
            img = np.expand_dims(img, axis = 0)

            return img
        except Exception as e:
            return str(e)

    
    def prediction(self, img_path):

        img = self.preprocessing_image(img_path)
        
        if not isinstance(img, str):
            pred = self.model.predict(img)[0]
            if pred[0] > 0.5:
                result = "Not Safe for Work (NSFW)"
            else:
                result = "Safe for Work (SFW)"

            return result
        else:
            return img