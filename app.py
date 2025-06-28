import streamlit as st
import numpy as np
import scipy.io
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
import mne_features.univariate as mne_f
import pickle

# Helper function to load MATLAB file
def load_mat_file(uploaded_file):
    try:
        return scipy.io.loadmat(uploaded_file)
    except Exception as e:
        st.error(f"Error loading .mat file: {e}")
        return None


def load_model(model_filename):
    try:
        with open(model_filename, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        print(f"Error loading {model_filename}: {e}")
        return None

# Feature extraction functions
def extract_features(data):
    n_trials, n_secs, n_channels, sfreq = data.shape
    freq_bands = [0.5, 4, 8, 13, 30, 50]

    ts_features, fb_features, hj_features, fr_features, ent_features = [], [], [], [], []

    for trial in data:
        for second in trial:
            # Time-Series features
            ts = np.concatenate([mne_f.compute_variance(second),
                                 mne_f.compute_rms(second),
                                 mne_f.compute_ptp_amp(second)])
            # Frequency Band features
            fb = mne_f.compute_pow_freq_bands(sfreq, second, freq_bands=freq_bands)
            # Hjorth features
            hj = np.concatenate([mne_f.compute_hjorth_mobility_spect(sfreq, second),
                                 mne_f.compute_hjorth_complexity_spect(sfreq, second)])
            # Fractal features
            fr = np.concatenate([mne_f.compute_higuchi_fd(second),
                                 mne_f.compute_katz_fd(second)])
            # Entropy features
            ent = np.concatenate([mne_f.compute_app_entropy(second),
                                  mne_f.compute_samp_entropy(second),
                                  mne_f.compute_spect_entropy(sfreq, second),
                                  mne_f.compute_svd_entropy(second)])

            ts_features.append(ts)
            fb_features.append(fb)
            hj_features.append(hj)
            fr_features.append(fr)
            ent_features.append(ent)

    combined_features = np.hstack([np.vstack(ts_features),
                                   np.vstack(fb_features),
                                   np.vstack(hj_features),
                                   np.vstack(fr_features),
                                   np.vstack(ent_features)])
    return ts_features, fb_features, hj_features, fr_features, ent_features, combined_features


# Plotting function for individual feature types
def plot_individual_features(ts_features, fb_features, hj_features, fr_features, ent_features):
    feature_titles = ['Time-Series Features', 'Frequency Bands Features', 'Hjorth Features',
                      'Fractal Features', 'Entropy Features']
    features = [ts_features, fb_features, hj_features, fr_features, ent_features]

    for i, (feature_list, title) in enumerate(zip(features, feature_titles)):
        plt.figure(figsize=(12, 6))
        for feature in feature_list:
            plt.plot(feature)
        plt.title(f"{title} (Individual Trials/Seconds)")
        plt.xlabel('Samples')
        plt.ylabel('Feature Value')
        plt.grid(True)
        st.pyplot(plt)


# Model prediction function for SVM, Random Forest, and MLP
def predict_with_model(model, features):
    try:
        predictions = model.predict(features)
        stress_labels = np.argmax(predictions, axis=1) + 1
        avg_stress = np.mean(stress_labels)
        max_stress = np.max(stress_labels)
        return stress_labels, avg_stress, max_stress
    except Exception as e:
        st.error(f"Error predicting with model: {e}")
        return None, None, None


# Main function
def preprocess_and_predict(uploaded_file):
    mat_data = load_mat_file(uploaded_file)
    if mat_data is None:
        return

    if 'Clean_data' not in mat_data:
        st.error("The uploaded .mat file does not contain the key 'Clean_data'.")
        return

    data = mat_data['Clean_data']
    st.write(f"Shape of Clean_data: {data.shape}")

    if data.shape != (32, 3200):
        st.error(f"Unexpected shape of Clean_data: {data.shape}. Expected (32, 3200).")
        return

    reshaped_data = data.reshape(1, 25, 32, 128)
    st.write(f"Reshaped data: {reshaped_data.shape}")

    # Extract features
    ts_features, fb_features, hj_features, fr_features, ent_features, combined_features = extract_features(
        reshaped_data)
    st.write("Combined Features Shape:", combined_features.shape)

    # Plot each feature type individually
    plot_individual_features(ts_features, fb_features, hj_features, fr_features, ent_features)

    # Load and predict with each model


    # CNN-LSTM Model Prediction
    st.subheader("CNN-LSTM Model Predictions")
    try:
        cnn_lstm_model = tf.keras.models.load_model('Models/cnn_lstm.h5')
        reshaped_features = combined_features.reshape(combined_features.shape[0], combined_features.shape[1], 1)
        predictions = cnn_lstm_model.predict(reshaped_features)
        stress_labels = np.argmax(predictions, axis=1) + 1
        avg_stress = np.mean(stress_labels)
        max_stress = np.max(stress_labels)
        st.write(f"CNN-LSTM Model Predictions (Stress Levels for each second):", stress_labels)
        st.write(f"Average Stress (CNN-LSTM): {avg_stress:.2f}, Maximum Stress (CNN-LSTM): {max_stress}")
    except FileNotFoundError:
        st.warning("CNN-LSTM model file not found. Please check the file path.")
    except Exception as e:
        st.error(f"Error loading CNN-LSTM model: {e}")


# Streamlit UI
st.title("EEG Stress Level Prediction")
uploaded_file = st.file_uploader("Upload your .mat file", type=["mat"])
if uploaded_file is not None:
    preprocess_and_predict(uploaded_file)
