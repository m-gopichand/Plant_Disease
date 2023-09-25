import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
from PIL import Image
import base64


image_path="/home/dev/PycharmProjects/PlantDisease/bg1.jpg"
image_base64 = base64.b64encode(open(image_path, 'rb').read()).decode()
# Define the CSS for the container
styles = f'''
    <style>
        .stApp {{
            background-image: linear-gradient(rgba(0, 0, 0, 0.2), rgba(0, 0, 0, 0.2)), url("data:image/jpeg;base64,{image_base64}");
            background-size: cover;
        }}
        .stApp .element-container, .stApp .streamlit-button, .stApp .stMarkdown, .stApp .stText {{
            color: black;
            font-size: 28px;
        }}
        .stApp h1, .stApp h2, .stApp h3, .stApp h4 {{
            color: black;
        }}
        .stMarkdown, .stText {{
            background: linear-gradient(45deg, rgba(144, 238, 144, 0.5), rgba(255, 165, 0, 0.5));
            padding: 10px;
            border-radius: 5px;
            color: black;
            font-weight: bold;
        }}
        .sidebar .sidebar-content {{
            background-color: #f5f5f5;
        }}
        .stContainer {{
            background-color: rgba(255, 165, 0, 0.3);
            background-image: linear-gradient(to bottom right, rgba(255, 165, 0, 0.3), rgba(0, 128, 0, 0.8));
            border-radius: 10px;
            padding: 20px;
        }}
    </style>
'''
st.markdown(styles, unsafe_allow_html=True)

# List of plants
plants = [
    "Apple", "Cassava", "Cherry", "Chili", "Coffee", "Corn", "Cucumber",
    "Guava", "Grape", "Jamun", "Lemon", "Mango", "Peach", "Pepper",
    "Pomegranate", "Potato", "Soybean", "Strawberry", "Sugarcane",
    "Tea", "Tomato", "Wheat"
]
# Dropdown for plant selection in the sidebar
selected_plant = st.sidebar.selectbox("🌱 Select a plant type:", plants)

# Based on the selection, load the model and data
model_path = f"{selected_plant.lower()}_model.h5"
data_path = f"{selected_plant.lower()}_data.csv"

model = tf.keras.models.load_model(model_path)
info_df = pd.read_csv(data_path)

# Main app
st.title("Detectify 🌿")
st.write(
    """
    Welcome to the plant disease classifier! Select your plant type from the sidebar, 
    upload a picture, and the app will identify the disease (if any) 
    present in your plant.
    """
)
text_box_style = f'''
    <style>
        .stMarkdown, .stText {{
            background: linear-gradient(45deg, rgba(144, 238, 144, 0.5), rgba(255, 165, 0, 0.5));  /* Gradient from light green to light orange */
            padding: 10px;
            border-radius: 5px;
            color: black;  /* Set text color to black */
            font-weight: bold;  /* Set font weight to bold */
        }}
    </style>
'''
st.markdown(text_box_style, unsafe_allow_html=True)


uploaded_file = st.file_uploader("Upload an image of your plant", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Define the CSS for the container
    container_style = f'''
        <style>
            .stContainer {{
                background-color: rgba(255, 165, 0, 0.3);  /* Light orange with 30% opacity */
                background-image: linear-gradient(to bottom right, rgba(255, 165, 0, 0.3), rgba(0, 128, 0, 0.8));  /* Gradient from light orange to dark green */
                border-radius: 10px;
                padding: 20px;
            }}
        </style>
    '''
    st.markdown(container_style, unsafe_allow_html=True)

    # Your content within the container
    with st.container():
        col1, col2 = st.columns(2)

        with col1:
            st.image(uploaded_file, caption="Uploaded Image", width=170)  # Adjust width as needed

        with col2:
            st.write("Classifying...")

            # Preprocess the image and make predictions
            image = Image.open(uploaded_file).resize((224, 224))
            img_array = np.array(image)[np.newaxis, :]
            predictions = model.predict(img_array)
            predicted_class = np.argmax(predictions)

            # Fetch the row corresponding to the predicted class
            predicted_row = info_df.loc[predicted_class].squeeze()

            # Fetch details
            disease_name = predicted_row['Disease']
            description = predicted_row['Description']
            why_it_occurred = predicted_row['Why it occurred']
            preventive_measures = predicted_row['Prevention measures']
            recommended_steps = predicted_row['Recommended steps to cure the disease']

            # Display on Streamlit
            st.write(f"**Disease Identified**: {disease_name}")
            st.write(f"**Description**: {description}")

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Why it occurred:")
                st.write(why_it_occurred)
            with col2:
                st.subheader("Preventive measures:")
                st.write(preventive_measures)

            st.subheader("Recommended steps to cure:")
            st.write(recommended_steps)

# Add footer
st.write("---")
st.write(
    """
    Made with ❤️ by The Green Guardians
    """
)
