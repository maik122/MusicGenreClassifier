# src/predict.py
# Loaded by app.py at startup.

import numpy as np
import librosa
import joblib
# import keras
from src.config import RF_PATH, SCALER_PATH, LE_PATH, CNN_PATH


def load_models():
    """
    Load all saved models and preprocessors from disk.
    Called once at Streamlit startup.

    Returns
    -------
    rf     : RandomForestClassifier
    scaler : StandardScaler
    le     : LabelEncoder
    cnn    : Keras Model
    """
    rf     = joblib.load(RF_PATH)
    scaler = joblib.load(SCALER_PATH)
    le     = joblib.load(LE_PATH)
   # cnn    = keras.models.load_model(CNN_PATH)
    return rf, scaler, le


def extract_features(audio_path: str, duration: int = 30) -> np.ndarray:
    """
    Extract the same features as features_30_sec.csv from a raw audio file.

    Parameters
    ----------
    audio_path : str  — path to uploaded .wav or .mp3
    duration   : int  — seconds to analyse

    Returns
    -------
    features : np.ndarray of shape (1, n_features)
    """
    y, sr = librosa.load(audio_path, duration=duration, mono=True)

    features = []

    # length
    features.append(len(y))

    # chroma_stft
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features += [chroma.mean(), chroma.var()]

    # rms
    rms = librosa.feature.rms(y=y)
    features += [rms.mean(), rms.var()]

    # spectral_centroid
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    features += [cent.mean(), cent.var()]

    # spectral_bandwidth
    bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    features += [bw.mean(), bw.var()]

    # rolloff
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features += [rolloff.mean(), rolloff.var()]

    # zero_crossing_rate
    zcr = librosa.feature.zero_crossing_rate(y)
    features += [zcr.mean(), zcr.var()]

    # harmony, perceptr
    harmony, perceptr = librosa.effects.hpss(y)
    features += [harmony.mean(), harmony.var(), perceptr.mean(), perceptr.var()]

    # tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    features.append(float(tempo))

    # mfcc1 - mfcc20 (mean and var for each)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    for coeff in mfcc:
        features += [coeff.mean(), coeff.var()]

    return np.array(features).reshape(1, -1)


def predict_genre(audio_path: str, rf, scaler, le) -> dict:
    """
    Predict genre from a raw audio file using the Random Forest.

    Returns
    -------
    dict with keys: genre (str), probabilities (dict of genre -> float)
    """
    features = extract_features(audio_path)
    features_scaled = scaler.transform(features)

    genre = le.inverse_transform(rf.predict(features_scaled))[0]
    probs = rf.predict_proba(features_scaled)[0]
    prob_dict = {le.classes_[i]: float(probs[i]) for i in range(len(le.classes_))}

    return {"genre": genre, "probabilities": prob_dict}