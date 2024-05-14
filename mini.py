import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import librosa.display
import os

from sklearn.model_selection import train_test_split


drive_path = r'D:\min pro\AudioWAV-20240510T160508Z-001'
print(os.listdir(drive_path))
os.chdir(r'D:\min pro\AudioWAV-20240510T160508Z-001\AudioWAV')  # Change the current working directory

# Now you can use the 'path' variable to collect all audio filenames
path = r'D:\min pro\AudioWAV-20240510T160508Z-001\AudioWAV'
audio_path = []
audio_emotion = []



# Collect all the audio filenames in the variable 'directory_path'
directory_path = os.listdir(path)

for audio in directory_path:
    audio_path.append(os.path.join(path, audio)) 
    #audio_path.append(path + audio)
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
dataset = pd.concat([audio_path_dataset, emotion_dataset], axis= 1)
#print(len(dataset))
print(dataset.head())


# counting audio categorized by emotions
plt.figure(figsize=(6,6), dpi=80)
plt.title("Emotion Count", size=16)
plt.xlabel('Emotions', size = 12)
plt.ylabel('Count', size = 12)
sns.histplot(dataset.Emotions, color='#F19C0E')
#plt.show()


emotion_sad = dataset[dataset['Emotions']=='sad']['Path']
print(type(emotion_sad))

# Define the index you want to access
index = 2

# Check the size of the array
array_size = len(emotion_sad.values)

# Ensure that the index is within bounds
if 0 <= index < array_size:
    data_path = emotion_sad.values[index]
    data, sampling_rate = librosa.load(data_path)
    # Further processing with the loaded data
else:
    print("Index is out of bounds.")



#choosing a file to plot wave and spectogram
#print(emotion_sad.values[65])
data_path = emotion_sad.values[542]
data, sampling_rate = librosa.load(data_path)
     
plt.figure(figsize=(10,6))
plt.title("Waveplot for a particular audio representing SAD emotion", size=16)
librosa.display.waveshow(data, sr=sampling_rate)
#plt.show()

plt.figure(figsize=(10,4))
plt.title("Spectogram for a particular audio representing SAD emotion", size=16)
D = librosa.stft(data)
S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
librosa.display.specshow(S_db, sr = sampling_rate, x_axis='time', y_axis='hz')
#plt.show()

# for audio processing accuracy
# add noise to audio and check how the waveplot changes
# also the observing the change in audio quality

## Augmentation (Noise Injection)
noise_amp = 0.035*np.random.uniform()*np.amax(data)
audio_injected_data = data + noise_amp*np.random.normal(size=data.shape[0])

# waveplot view after noise injection:
plt.figure(figsize=(10,6))
plt.title("Waveplot for a particular audio representing SAD emotion after noise injection", size=16)
librosa.display.waveshow(audio_injected_data, sr=sampling_rate)
plt.show()

X, Y = [], []
print("Feature processing...")

for path, emo, index in zip(dataset.Path, dataset.Emotions, range(len(dataset))):
    # Skip non-audio files
    if not path.endswith('.wav'):
        continue

    try:
        value, sample = librosa.load(path)
        # noise injection
        noise_amp = 0.035 * np.random.uniform() * np.amax(value)
        value = value + noise_amp * np.random.normal(size=value.shape[0])
        # mfcc
        mfcc = librosa.feature.mfcc(y=value, sr= sample, n_mfcc=13, n_fft=200, hop_length=512)
        mfcc = np.ravel(mfcc.T)
        # mel
        mel = librosa.feature.melspectrogram(y=value, sr=sample, hop_length = 256, n_fft = 512, n_mels=64)
        mel = librosa.power_to_db(mel ** 2)
        mel = np.ravel(mel).T
        result = np.array([])
        result = np.hstack((result, mfcc, mel))
        result = np.array(result)
        X.append(result)
        Y.append(emo)
    except Exception as e:
        print(f"Error processing {path}: {e}")
        continue

print("Feature processing completed.")

# print(X)
# print(Y)
extracted_audio_df = pd.DataFrame(X)
extracted_audio_df["emotion_of_audio"] = Y
print(extracted_audio_df.shape)
print(extracted_audio_df.tail(10))
extracted_audio_df = extracted_audio_df.fillna(0)
#print(extracted_audio_df.isna().any())


# preparing to train
X = extracted_audio_df.drop(labels='emotion_of_audio', axis= 1)
Y = extracted_audio_df['emotion_of_audio']

x_train, x_test, y_train, y_test = train_test_split(np.array(X), Y, test_size=0.2)

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


mlp_model = MLPClassifier(activation='relu',
                         solver='sgd',
                         hidden_layer_sizes=100,
                         alpha=0.839903176695813,
                         batch_size=150,
                         learning_rate='adaptive',
                         max_iter=100000)
# Fit mlp model
mlp_model.fit(x_train,y_train)

y_pred = mlp_model.predict(x_test)
accuracy=accuracy_score(y_true=y_test, y_pred=y_pred)

# the accuracy didn't turn out to be that good :(
print("\nModel:{}    Accuracy: {:.2f}%".
          format(type(mlp_model).__name__ , accuracy*100))


# the prediction made by the model:
print("The Prediction Made By Model: ")
print("<<<===========================================>>>")
df = pd.DataFrame({'Actual': y_test, 'Predict': y_pred})
print(df.head())
user_audio_path = input("Please provide the file path of the audio you want to analyze: ")
try:
    user_data, user_sampling_rate = librosa.load(user_audio_path)
    # Apply the same preprocessing steps as done for the dataset audios
    # For example, noise injection
    noise_amp = 0.035 * np.random.uniform() * np.amax(user_data)
    user_data = user_data + noise_amp * np.random.normal(size=user_data.shape[0])
    # Extract features
    user_mfcc = librosa.feature.mfcc(y=user_data, sr=user_sampling_rate, n_mfcc=13, n_fft=200, hop_length=512)
    user_mfcc = np.ravel(user_mfcc.T)
    user_mel = librosa.feature.melspectrogram(y=user_data, sr=user_sampling_rate, hop_length=256, n_fft=512, n_mels=64)
    user_mel = librosa.power_to_db(user_mel ** 2)
    user_mel = np.ravel(user_mel).T
    user_features = np.hstack((user_mfcc, user_mel))
    user_features = np.array(user_features).reshape(1, -1)
    # Scale the features
    user_features = scaler.transform(user_features)
except Exception as e:
    print("Error processing user audio:", e)
    exit()
try:
    user_emotion_prediction = mlp_model.predict(user_features)
except Exception as e:
    print("Error predicting user emotion:", e)
    exit()
