import streamlit as st
from utils import load_cnn_model, preprocess_image, predict_seizure, is_valid_eeg_image

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
    st.title("🧠 Seizure Prediction")
    st.subheader(f"Welcome {st.session_state.username}! \nUpload an image to predict seizure likelihood.")

    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded Image")
        
        if not is_valid_eeg_image(uploaded_file):
            st.error("⚠️ This doesn't look like a valid EEG image. Please try another file.")
        else:
            with st.spinner("Predicting..."):
                model = load_cnn_model()
                processed_img = preprocess_image(uploaded_file)
                result = predict_seizure(model, processed_img)
            st.success(f"Prediction: **{result}**")

    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.rerun()

# Run the correct interface
if st.session_state.logged_in:
    app_interface()
else:
    login_form()
