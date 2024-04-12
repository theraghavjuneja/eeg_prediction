import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import mne

# Load the model
model = load_model("eeg_model.h5")

# Define a function to preprocess the EEG data
def preprocess_eeg(file_path):
    # Load the EEG data from the .fif file
    epochs = mne.read_epochs(file_path)
    # Extract data
    data = epochs.get_data()
    return data

# Function to get the label corresponding to the maximum value
def get_max_label(row):
    labels = ["t0", "t1", "t2"]
    max_index = np.argmax(row)
    return labels[max_index]

# Streamlit app
def main():
    st.title("EEG Data Prediction")

    # File uploader
    st.sidebar.title("Upload EEG Data")
    file = st.sidebar.file_uploader("Upload .fif file", type=["fif"])

    if file is not None:
        # Preprocess the uploaded EEG data
        eeg_data = preprocess_eeg(file)
        # Make predictions
        predictions = model.predict(eeg_data)

        # Display predictions
        st.subheader("Predictions")
        for i, row in enumerate(predictions):
            st.write(f"Row {i+1}: {get_max_label(row)}")

if __name__ == "__main__":
    main()
