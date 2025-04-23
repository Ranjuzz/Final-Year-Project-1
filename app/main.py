import streamlit as st
import numpy as np
from utils import load_cnn_model, predict_from_npz

# Dummy credentials
USER_CREDENTIALS = {
    "admin": "1234",
    "user": "1234"
}

# Load styles
with open("app/styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Initialize session state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""

def login_form():
    st.title("🔐 Login to Seizure Prediction App")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")

        if submit:
            if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success(f"Welcome, {username}!")
                st.rerun()
            else:
                st.error("Invalid username or password")

def app_interface():
    st.title("🧠 Epilepsy Prediction")
    st.subheader(f"Welcome {st.session_state.username}! \nUpload a `.npz` EEG file for prediction.")

    uploaded_file = st.file_uploader("Choose an EEG file (.npz)", type=["npz"])
    if uploaded_file:
        try:
            with st.spinner("Predicting..."):
                npz_data = np.load(uploaded_file)
                model = load_cnn_model()
                result = predict_from_npz(model, npz_data)
            st.success(f"Prediction: **{result}**")
        except Exception as e:
            st.error(f"Error processing EEG file: {e}")

    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.rerun()

# Run the correct interface
if st.session_state.logged_in:
    app_interface()
else:
    login_form()
