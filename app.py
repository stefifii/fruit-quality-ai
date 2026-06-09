import streamlit as st
import cv2
import numpy as np
import pyttsx3
from tensorflow.keras.models import load_model

# -----------------------------------
# PAGE SETTINGS
# -----------------------------------

st.set_page_config(
    page_title="Fruit Quality AI",
    page_icon="🍎",
    layout="centered"
)

# -----------------------------------
# CUSTOM CSS
# -----------------------------------

st.markdown("""
<style>

.stApp {
    background-color: #0E1117;
    color: white;
}

h1 {
    text-align: center;
    color: #FF4B4B;
}

.stButton>button {
    background-color: #FF4B4B;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------------
# LOAD MODEL
# -----------------------------------

model = load_model("fruit_quality_model.h5")

engine = pyttsx3.init()

# -----------------------------------
# CLASS LABELS
# -----------------------------------

class_labels = {
    0: 'Fresh Apple',
    1: 'Fresh Banana',
    2: 'Rotten Apple',
    3: 'Rotten Banana'
}

# -----------------------------------
# SIDEBAR
# -----------------------------------

st.sidebar.title("📊 Model Information")

st.sidebar.info("""
This AI model detects:

✅ Fresh Apple  
✅ Fresh Banana  
❌ Rotten Apple  
❌ Rotten Banana  

Built using:
- TensorFlow
- CNN
- Streamlit
""")

# -----------------------------------
# TITLE
# -----------------------------------

st.title("🍎 Fruit Quality Detection AI")

st.write("Upload an image or use webcam for prediction.")

# -----------------------------------
# FILE UPLOAD
# -----------------------------------

uploaded_file = st.file_uploader(
    "Upload Fruit Image",
    type=["jpg", "jpeg", "png"]
)

# -----------------------------------
# CAMERA INPUT
# -----------------------------------

camera_image = st.camera_input("Take a Picture")

# -----------------------------------
# PREDICTION FUNCTION
# -----------------------------------

def predict_image(image_file):

    # Convert image to bytes
    file_bytes = np.asarray(
        bytearray(image_file.read()),
        dtype=np.uint8
    )

    # Decode image
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Flip webcam image
    img = cv2.flip(img, 1)

    # Display image
    st.image(img, caption="Selected Image")

    # Resize image
    resized_img = cv2.resize(img, (100, 100))

    # Normalize image
    resized_img = resized_img.astype("float32") / 255.0

    # Expand dimensions
    resized_img = np.expand_dims(resized_img, axis=0)

    # Predict
    prediction = model.predict(resized_img)

    predicted_index = np.argmax(prediction)

    predicted_class = class_labels[predicted_index]

    confidence = np.max(prediction) * 100

    # -----------------------------------
    # RESULTS
    # -----------------------------------

    st.subheader("🔍 Prediction Result")

    if "Fresh" in predicted_class:
        st.success(f"✅ {predicted_class}")
        st.balloons()
        st.success("Fruit is Fresh and Safe to Eat")

        engine.say(f"The prediction is {predicted_class}")

        engine.runAndWait()

    else:
        st.error(f"❌ {predicted_class}")
        st.warning("Fruit may be Spoiled")

        engine.say(f"The prediction is {predicted_class}")

        engine.runAndWait()

    st.info(f"Confidence: {confidence:.2f}%")

    st.progress(int(confidence))

    # -----------------------------------
    # ALL PROBABILITIES
    # -----------------------------------

    st.subheader("📈 Prediction Probabilities")

    for i, prob in enumerate(prediction[0]):
        st.write(f"{class_labels[i]} : {prob*100:.2f}%")

# -----------------------------------
# RUN PREDICTION
# -----------------------------------

if uploaded_file is not None:
    predict_image(uploaded_file)

elif camera_image is not None:
    predict_image(camera_image)