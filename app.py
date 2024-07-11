from flask import Flask, request, jsonify
import numpy as np
import librosa
import tensorflow as tf

app = Flask(__name__)

# Load your pre-trained model
model_path = 'C:/Users/Kingdom City/Desktop/Folder/my_flask_project/CNN.h5'
model = tf.keras.models.load_model(model_path)

def preprocess_audio(audio_file):
    y, sr = librosa.load(audio_file, sr=None)
    # Extracting Mel spectrogram
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
    spectrogram_db_scaled = np.mean(spectrogram_db.T, axis=0)
    return spectrogram_db_scaled

@app.route('/check', methods=['POST'])
def check_file_condition():
    file_path = request.json.get('file_path')
    condition = request.json.get('condition')
    
    if file_path == "C:/Users/Kingdom City/Desktop/Audio files of brain waves/Spectograms/Beta2.png" and condition == "Anxiety":
        return jsonify({'message': 'C:/Users/Kingdom City/Desktop/Binarual beats/Happy music/Alpha Binaurual.wav'})
    else:
        return jsonify({'message': 'Condition not met'}), 400

if __name__ == '__main__':
    app.run(debug=True)
    