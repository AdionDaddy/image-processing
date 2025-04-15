import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import io

# Set page config
st.set_page_config(
    page_title="Lung Cancer CT Scan Classifier",
    page_icon="🫁",
    layout="wide"
)

# Title and description
st.title("Lung Cancer CT Scan Classification")
st.markdown("""
This application uses a pre-trained ResNet50 model to classify chest CT scans into four categories:
- Normal (No Cancer)
- Adenocarcinoma
- Squamous Cell Carcinoma
- Large Cell Carcinoma
""")

# Function to load the model
@st.cache_resource
def load_classification_model():
    try:
        # Placeholder for model loading - replace with your actual model path
        # model = load_model('resnet50_lung_cancer_model.h5')
        # For demonstration, we'll use a placeholder message since the actual model isn't accessible
        return "Model would be loaded here"
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Function to preprocess the image
def preprocess_image(img):
    # Resize image to 224x224 as used in training
    img = img.resize((224, 224))
    # Convert to array and expand dimensions
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    # Return preprocessed image
    return img_array

# Function to make prediction
def predict(model, img_array):
    # This is a placeholder function
    # In a real implementation, you would use:
    # predictions = model.predict(img_array)
    
    # For demonstration, return random probabilities
    import random
    classes = ['Normal', 'Adenocarcinoma', 'Squamous Cell Carcinoma', 'Large Cell Carcinoma']
    # Simulating prediction probabilities
    probs = [random.uniform(0.0, 1.0) for _ in range(4)]
    # Normalize to sum to 1
    probs = [p/sum(probs) for p in probs]
    return dict(zip(classes, probs))

# Sidebar information
with st.sidebar:
    st.header("About")
    st.info("""
    This application demonstrates how a deep learning model can be used to classify lung cancer types from CT scans.
    
    **Note:** In a production environment, this would connect to a properly trained and validated medical model. 
    The current implementation is for demonstration purposes only.
    """)
    
    st.header("Instructions")
    st.markdown("""
    1. Upload a chest CT scan image using the file uploader
    2. The model will classify the image
    3. View the prediction results and confidence scores
    """)
    
    st.header("Dataset Information")
    st.markdown("""
    The model was trained on a dataset of lung CT scans containing four classes:
    - Normal: CT scans showing no signs of lung cancer
    - Adenocarcinoma: A type of non-small cell lung cancer
    - Squamous Cell Carcinoma: A type of non-small cell lung cancer that starts in the flat cells
    - Large Cell Carcinoma: A type of non-small cell lung cancer that tends to grow and spread quickly
    """)

# Main content
st.header("Upload a CT Scan Image")
uploaded_file = st.file_uploader("Choose a chest CT scan image...", type=["jpg", "jpeg", "png"])

# Placeholder for the model (in a real app, you'd load the actual model)
model = load_classification_model()

# Process the uploaded file
if uploaded_file is not None:
    # Display the uploaded image
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Uploaded Image")
        img = Image.open(uploaded_file)
        st.image(img, width=350)
    
    # Preprocess the image
    img_array = preprocess_image(img)
    
    # Make prediction
    with st.spinner('Analyzing CT scan...'):
        # Simulate processing time
        import time
        time.sleep(2)
        
        prediction = predict(model, img_array)
    
    # Display results
    with col2:
        st.subheader("Prediction Results")
        
        # Find the class with highest probability
        prediction_class = max(prediction, key=prediction.get)
        
        # Display the result with color coding
        if prediction_class == 'Normal':
            st.success(f"Prediction: **{prediction_class}**")
        else:
            st.error(f"Prediction: **{prediction_class}**")
        
        # Display confidence scores
        st.subheader("Confidence Scores")
        
        # Create a bar chart of prediction probabilities
        fig, ax = plt.subplots(figsize=(8, 4))
        
        classes = list(prediction.keys())
        scores = list(prediction.values())
        
        bars = ax.barh(classes, scores, color=['green', 'red', 'red', 'red'])
        
        # Add percentage labels to the bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{width:.1%}', ha='left', va='center')
        
        ax.set_xlim(0, 1.1)
        ax.set_xlabel('Probability')
        ax.set_title('Prediction Confidence Scores')
        
        # Display the chart
        st.pyplot(fig)
        
        # Warning message
        st.warning("""
        **Medical Disclaimer:** This tool is for demonstration purposes only and should not be used for actual medical diagnosis. 
        Always consult with healthcare professionals for proper medical advice and diagnosis.
        """)

else:
    # Display sample images when no file is uploaded
    st.info("Please upload a chest CT scan image to get started.")
    
    # Display sample images of each class
    st.subheader("Sample Images from Each Class")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("**Normal**")
        st.image("https://via.placeholder.com/224x224.png?text=Normal+Sample", width=150)
        
    with col2:
        st.markdown("**Adenocarcinoma**")
        st.image("https://via.placeholder.com/224x224.png?text=Adenocarcinoma+Sample", width=150)
        
    with col3:
        st.markdown("**Squamous Cell Carcinoma**")
        st.image("https://via.placeholder.com/224x224.png?text=Squamous+Sample", width=150)
        
    with col4:
        st.markdown("**Large Cell Carcinoma**")
        st.image("https://via.placeholder.com/224x224.png?text=Large+Cell+Sample", width=150)

# Add footer
st.markdown("---")
st.markdown("""
<div style="text-align: center">
    <p>Lung Cancer CT Scan Classification Tool | Built with Streamlit and TensorFlow</p>
</div>
""", unsafe_allow_html=True)