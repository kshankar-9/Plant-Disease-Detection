
import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image

st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌱",
    layout="centered"
)

MODEL_PATH = "plant_disease_model(1).keras"

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_data
def load_classes():
    with open("class_names.json", "r") as f:
        return json.load(f)

model = load_model()
class_names = load_classes()

st.title("🌱 Plant Disease Detection")
st.write(
    "Upload a clear plant leaf image to predict "
    "its disease using MobileNetV2."
)

uploaded_file = st.file_uploader(
    "Choose a leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Leaf", use_container_width=True)

    if st.button("🔍 Predict Disease"):
        with st.spinner("Analyzing leaf image..."):

            img = image.resize((160, 160))
            img_array = np.array(img, dtype=np.float32)

            img_array = tf.keras.applications.mobilenet_v2.preprocess_input(
                img_array
            )

            img_array = np.expand_dims(img_array, axis=0)

            predictions = model.predict(img_array, verbose=0)[0]

            predicted_index = int(np.argmax(predictions))
            confidence = float(predictions[predicted_index]) * 100

            disease = class_names[predicted_index]

            st.subheader("Prediction Result")

            if confidence < 70:
                st.warning(
                    "Image unclear or prediction uncertain. "
                    "Please upload a clear leaf image."
                )
            else:
                plant, condition = disease.split("___", 1)

                st.success("Prediction completed!")

                st.write("**Plant:**", plant.replace("_", " "))
                st.write(
                    "**Predicted Condition:**",
                    condition.replace("_", " ")
                )

                st.metric(
                    "Model Confidence",
                    f"{confidence:.2f}%"
                )

                if "healthy" in condition.lower():
                    st.info(
                        "The model predicts a healthy plant leaf."
                    )
                else:
                    st.warning(
                        "The model predicts a possible plant disease."
                    )

                st.write("**Top 3 predictions**")

                top_indices = np.argsort(predictions)[-3:][::-1]

                for index in top_indices:
                    st.write(
                        f"{class_names[index].replace('___', ' - ')}: "
                        f"{predictions[index] * 100:.2f}%"
                    )

                st.caption(
                    "Educational AI prediction only. "
                    "Not a confirmed agricultural diagnosis."
                )
              
