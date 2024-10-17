import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Configurar Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploaded_images'

# Cargar el modelo de gatos vs. perros
model = tf.keras.models.load_model('cats_vs_dogs_model.h5')

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalización

    # Hacer predicción
    prediction = model.predict(img_array)
    return 'Dog' if prediction[0] > 0.5 else 'Cat'

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)

        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Predecir si es perro o gato
            prediction = predict_image(filepath)

            return render_template('index.html', prediction=prediction, image_path=filepath)

    return render_template('index.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)
