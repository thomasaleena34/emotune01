from flask import Flask, render_template, request, jsonify
import os
import warnings
import pandas as pd
import numpy as np
import librosa
import librosa.display
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from flask_cors import CORS

warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Define the emotion detection function using the trained model
def detect_emotion_from_file(file_path):
    try:
        data, sampling_rate = librosa.load(file_path)
        mfcc = librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=13, n_fft=200, hop_length=512)
        mfcc = np.ravel(mfcc.T)
        mel = librosa.feature.melspectrogram(y=data, sr=sampling_rate, hop_length=256, n_fft=512, n_mels=64)
        mel = librosa.power_to_db(mel ** 2)
        mel = np.ravel(mel).T
        features = np.hstack((mfcc, mel))
        features = features.reshape(1, -1)
        features = scaler.transform(features)
        emotion = mlp_model.predict(features)[0]
        return emotion
    except Exception as e:
        print(f"Error detecting emotion from file {file_path}: {e}")
        return "unknown"

# Emotion detection endpoint
@app.route('/detect-emotion', methods=['POST'])
def detect_emotion():
    if 'voice' not in request.files:
        return jsonify({"error": "No file part"}), 400
    voice_data = request.files['voice']
    if voice_data.filename == '':
        return jsonify({"error": "No selected file"}), 400

    voice_file_path = os.path.join('uploads', voice_data.filename)
    voice_data.save(voice_file_path)
    detected_emotion = detect_emotion_from_file(voice_file_path)
    os.remove(voice_file_path)
    return jsonify({"emotion": detected_emotion})

# Playlist retrieval endpoint
@app.route('/get-playlist', methods=['GET'])
def get_playlist():
    emotion = request.args.get('emotion')
    playlists = {
        "happy": "https://example.com/happy-playlist.mp3",
        "sad": "https://example.com/sad-playlist.mp3",
        "angry": "https://example.com/angry-playlist.mp3",
    }
    playlist_url = playlists.get(emotion, "https://example.com/default-playlist.mp3")
    return jsonify({"playlist_url": playlist_url})

# Homepage
@app.route('/')
def index():
    return render_template('html1.html')

if __name__ == '__main__':
    # Load and preprocess the dataset
    drive_path = r'D:\min pro\AudioWAV-20240510T160508Z-001'
    path = os.path.join(drive_path, 'AudioWAV')
    audio_path = []
    audio_emotion = []

    directory_path = os.listdir(path)
    for audio in directory_path:
        audio_path.append(os.path.join(path, audio))
        emotion = audio.split('_')
        if len(emotion) >= 3:
            if emotion[2] == 'SAD':
                audio_emotion.append("sad")
            elif emotion[2] == 'ANG':
                audio_emotion.append("angry")
            elif emotion[2] == 'DIS':
                audio_emotion.append("disgust")
            elif emotion[2] == 'NEU':
                audio_emotion.append("neutral")
            elif emotion[2] == 'HAP':
                audio_emotion.append("happy")
            elif emotion[2] == 'FEA':
                audio_emotion.append("fear")
            else:
                audio_emotion.append("unknown")
        else:
            audio_emotion.append("unknown")

    emotion_dataset = pd.DataFrame(audio_emotion, columns=['Emotions'])
    audio_path_dataset = pd.DataFrame(audio_path, columns=['Path'])
    dataset = pd.concat([audio_path_dataset, emotion_dataset], axis=1)

    X, Y = [], []
    for path, emo in zip(dataset.Path, dataset.Emotions):
        if not path.endswith('.wav'):
            continue
        try:
            value, sample = librosa.load(path)
            noise_amp = 0.035 * np.random.uniform() * np.amax(value)
            value = value + noise_amp * np.random.normal(size=value.shape[0])
            mfcc = librosa.feature.mfcc(y=value, sr=sample, n_mfcc=13, n_fft=200, hop_length=512)
            mfcc = np.ravel(mfcc.T)
            mel = librosa.feature.melspectrogram(y=value, sr=sample, hop_length=256, n_fft=512, n_mels=64)
            mel = librosa.power_to_db(mel ** 2)
            mel = np.ravel(mel).T
            features = np.hstack((mfcc, mel))
            X.append(features)
            Y.append(emo)
        except Exception as e:
            print(f"Error processing {path}: {e}")
            continue

    extracted_audio_df = pd.DataFrame(X)
    extracted_audio_df["emotion_of_audio"] = Y
    extracted_audio_df = extracted_audio_df.fillna(0)

    X = extracted_audio_df.drop(labels='emotion_of_audio', axis=1)
    Y = extracted_audio_df['emotion_of_audio']

    x_train, x_test, y_train, y_test = train_test_split(np.array(X), Y, test_size=0.2)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    mlp_model = MLPClassifier(activation='relu',
                              solver='sgd',
                              hidden_layer_sizes=(100,),
                              alpha=0.839903176695813,
                              batch_size=150,
                              learning_rate='adaptive',
                              max_iter=100000)
    mlp_model.fit(x_train, y_train)
    y_pred = mlp_model.predict(x_test)
    accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
    print(f"Model: {type(mlp_model).__name__}    Accuracy: {accuracy * 100:.2f}%")

    app.run(debug=True)
