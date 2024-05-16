from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from PIL import Image
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd
import os

app = Flask(__name__)
# Define the initial counter value
upload_counter = 1
# Define the folder where uploaded images will be stored
UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define allowed image file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

@app.route('/')
def home():
    return render_template('index.html')


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/upload', methods=['POST'])
def upload_image():
    global upload_counter  # Access the global counter
    # Check if a file was included in the request
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    # Check if the file has a valid file extension
    if file and allowed_file(file.filename):
        # Generate a unique filename for the uploaded image
        unique_filename = f"{upload_counter:04d}_{secure_filename(file.filename)}"
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], unique_filename))
        cat= compute_result(unique_filename)
        upload_counter += 1  # Increment the counter
        return jsonify({'message': 'File uploaded successfully', 'filename': unique_filename, "Disease: ": cat})
    else:
        return jsonify({'error': 'File Upload Not Successful'})

def compute_result(filename):
    model = keras.models.load_model('citrus.h5')
    # Making a prediction
    test_image=r'uploads/'+filename
    image_result=Image.open(test_image)
    test_image=image.load_img(test_image,target_size=(224,224))
    test_image=image.img_to_array(test_image)
    test_image=test_image/255
    test_image=np.expand_dims(test_image,axis=0)
    result=model.predict(test_image)
    print(np.argmax(result))
    Categories=['healthy','multiple_disease','rust','scab']
    return Categories[np.argmax(result)]

if __name__ == '__main__':
    app.run(debug=True)