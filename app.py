from flask import Flask, request, render_template, jsonify
from tensorflow import keras 
from keras.models import load_model
#from keras.preprocessing.image import img_to_array, load_img
from keras.utils import img_to_array, load_img
import numpy as np
import os

app = Flask(__name__)

# Load the model
model = load_model('model/MobileNet.keras')

# Define a function to process the image
def prepare_image(image, target_size):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if an image is uploaded
        if 'file' not in request.files:
            return render_template('index.html', prediction='No file part')
        
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', prediction='No selected file')

        if file:
            image = load_img(file, target_size=(224, 224))  # adjust the target size as per your model's input
            image = prepare_image(image, target_size=(224, 224))

            # Make prediction
            prediction = model.predict(image)
            # Interpret the prediction based on your model's output
            disease_prediction = np.argmax(prediction, axis=1)[0]
            diseases = ['Disease A', 'Disease B', 'Disease C']
            result = diseases[disease_prediction]

            return render_template('index.html', prediction=result)

    return render_template('index.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)
