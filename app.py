from flask import Flask, request, jsonify
import numpy as np
import librosa
import tensorflow as tf

app = Flask(__name__)

# Load your pre-trained model
model = tf.keras.models.load_model('C:\Users\Kingdom City\Desktop\Folder\my_flask_project\CNN.h5')

def preprocess_audio(audio_file):
    y, sr = librosa.load(audio_file, sr=None)
    # Extracting Mel spectrogram
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
    spectrogram_db_scaled = np.mean(spectrogram_db.T, axis=0)
    return spectrogram_db_scaled

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Read the audio file from the request
        audio_data = io.BytesIO(file.read())
        features = preprocess_audio(audio_data)
        features = np.expand_dims(features, axis=0)
        
        # Make prediction
        prediction = model.predict(features)
        response = {'prediction': prediction.tolist()}
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)