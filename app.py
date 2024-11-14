import os
import numpy as np
from flask import Flask, render_template, request
from PIL import Image
from tensorflow.keras.models import load_model

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model_path = r"C:\Users\Veena Yadav\PycharmProjects\pythonProject3\model\skin_cancer_detection_model.h5"
model = load_model(model_path)

# Define class labels (update this according to your model's output)
class_labels = [
    'melanoma',
    'seborrheic keratosis',
    'basal cell carcinoma',
    'squamous cell carcinoma',
    'actinic keratosis',
    'dermatofibroma',
    'vascular lesion',
    'nevus',
    'benign keratosis'
]


@app.route('/', methods=['GET', 'POST'])
def index():
    result = None  # Initialize result variable
    if request.method == 'POST':
        # Handle image upload
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            # Save the uploaded file temporarily
            file_path = os.path.join('uploads', uploaded_file.filename)
            uploaded_file.save(file_path)

            # Preprocess the image
            img = Image.open(file_path).convert('L')  # Convert to grayscale
            img = img.resize((224, 224))  # Resize to match model input
            img_array = np.array(img) / 255.0  # Normalize the image

            # Convert grayscale to RGB by repeating the channels
            img_array = np.stack((img_array,) * 3, axis=-1)  # Convert grayscale to RGB
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

            # Predict the class
            predictions = model.predict(img_array)
            predicted_index = np.argmax(predictions, axis=1)[0]
            result = class_labels[predicted_index]

            # Clean up by removing the uploaded file if needed
            os.remove(file_path)

    return render_template('index.html', result=result)


if __name__ == '__main__':
    app.run(debug=True)
