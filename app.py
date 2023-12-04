from flask import Flask, render_template, request
from keras.preprocessing.image import load_img
from flask import Flask, render_template, request, flash
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.applications.nasnet import preprocess_input
import numpy as np


app = Flask(__name__)
model = load_model('C:/xampp/htdocs/image-classifier-main/NasNetMobile_coffee-Jenis Kopi-99.78.h5')

@app.route('/', methods=['GET'])
def hello_word():
    return render_template('index.html')

@app.route('/', methods=['POST'])
@app.route('/', methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    
    # Memeriksa apakah ada file yang diunggah
    if 'imagefile' not in request.files or imagefile.filename == '':
        flash('Silakan unggah gambar terlebih dahulu.')
        return render_template('index.html')
    
    image_path = "./images/" + imagefile.filename
    imagefile.save(image_path)

    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    
    predictions = model.predict(image)
    probabilities = np.squeeze(predictions)  # Memperoleh probabilitas untuk setiap kelas
    class_labels = ['ARABIKA', 'ROBUSTA']
    top_class_indices = np.argsort(probabilities)[::-1][:len(class_labels)]
    top_class_probs = probabilities[top_class_indices]

    prediction_results = {}
    for i in range(len(top_class_indices)):
        class_label = class_labels[top_class_indices[i]]
        class_probability = top_class_probs[i]
        prediction_results[class_label.lower() + '_prob'] = f'{class_probability:.2f}'

    return render_template('index.html', prediction=prediction_results, image_path=image_path)

if __name__ == '__main__':
    app.run(port=3000, debug=True)
