import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import joblib
import pickle
import nltk
import os
import warnings
import tensorflow as tf
from tensorflow.keras.applications.xception import preprocess_input
from PIL import Image
warnings.filterwarnings('ignore')

# Download NLTK data
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Streamlit app title and description
st.title("Fake News & Deepfake Detection App")
st.markdown("""
This app detects whether a news article is likely to be fake or real based on its title and text, 
and whether an uploaded image is a deepfake or real. It uses machine learning models trained on 
news articles (Logistic Regression) and images (Xception-based CNN).
""")

# Load fake news models and preprocessors
@st.cache_resource
def load_fake_news_models():
    try:
        model = joblib.load('fake_news_model.joblib')
        tfidf = joblib.load('tfidf_vectorizer.joblib')
        with open('selected_features.pkl', 'rb') as f:
            selected_features = pickle.load(f)
        return model, tfidf, selected_features
    except FileNotFoundError as e:
        st.error(f"Error: Fake news model or preprocessor files not found. Ensure 'fake_news_model.joblib', 'tfidf_vectorizer.joblib', and 'selected_features.pkl' are in the working directory.")
        st.stop()
        return None, None, None

# Load deepfake model
@st.cache_resource
def load_deepfake_model():
    try:
        model = tf.keras.models.load_model('deepfake_model.h5')
        return model
    except Exception as e:
        st.error(f"Error loading deepfake model: {e}. Ensure 'deepfake_model.h5' is in the working directory.")
        st.stop()
        return None

calibrated_model, tfidf, selected_features = load_fake_news_models()
deepfake_model = load_deepfake_model()

# Function to get base model from CalibratedClassifierCV
def get_base_model(calibrated_model):
    try:
        return calibrated_model.estimator
    except AttributeError as e:
        st.error(f"Error accessing base model: {e}")
        st.stop()
        return None

# --- Fake News Detection Section ---
st.header("Fake News Detection")
st.markdown("Enter a news article's title and text to predict if it's fake or real.")

# Input fields for fake news
title = st.text_input("Enter News Title", value="")
text = st.text_area("Enter News Text", value="", height=200)
threshold = 0.4  # Fixed threshold from notebook

# Function to preprocess fake news input
def preprocess_fake_news_input(title):
    try:
        title = title.lower().replace('news', '')
        tfidf_features = tfidf.transform([title]).toarray()
        return tfidf_features
    except Exception as e:
        st.error(f"Error preprocessing fake news input: {e}")
        st.stop()
        return None

# Function to generate feature importance plot
def plot_feature_importance(model, selected_features):
    try:
        base_model = get_base_model(model)
        if base_model is None:
            return None
        model_name = type(base_model).__name__

        fig, ax = plt.subplots(figsize=(8, 4))
        if model_name == 'LogisticRegression':
            if hasattr(base_model, 'coef_'):
                coefs = pd.Series(base_model.coef_[0], index=selected_features)
                coefs.sort_values().plot(kind='barh', ax=ax, color='skyblue')
                plt.title('Feature Importance (Logistic Regression Coefficients)')
                plt.xlabel('Coefficient Value')
            else:
                st.warning("LogisticRegression model does not have coefficients.")
                return None
        else:
            st.warning(f"Feature importance not available for {model_name}.")
            return None
        plt.tight_layout()
        return fig
    except Exception as e:
        st.error(f"Error generating feature importance plot: {e}")
        return None

# Fake news prediction
if st.button("Predict Fake News"):
    if title.strip() == "" or text.strip() == "":
        st.warning("Please enter both a title and text to make a fake news prediction.")
    else:
        # Preprocess input
        X_input = preprocess_fake_news_input(title)
        if X_input is None:
            st.stop()
        
        # Filter for selected features
        tfidf_feature_names = tfidf.get_feature_names_out()
        selected_indices = [i for i, name in enumerate(tfidf_feature_names) if name in selected_features]
        X_selected = X_input[:, selected_indices]
        
        # Make prediction
        try:
            y_pred_prob = calibrated_model.predict_proba(X_selected)[:, 1]
            y_pred = (y_pred_prob > threshold).astype(int)
            label = "Fake" if y_pred[0] == 1 else "Real"
            confidence = y_pred_prob[0] if label == "Fake" else 1 - y_pred_prob[0]
            
            # Display prediction
            st.subheader("Fake News Prediction Result")
            st.write(f"**Prediction**: {label}")
            st.write(f"**Confidence**: {confidence:.4f}")
        except Exception as e:
            st.error(f"Error making fake news prediction: {e}")
            st.stop()
        
        # Feature importance plot
        st.subheader("Feature Importance")
        fig = plot_feature_importance(calibrated_model, selected_features)
        if fig:
            st.pyplot(fig)
            st.markdown("This plot shows the importance of each feature based on the model's coefficients. Positive values favor 'Fake', negative favor 'Real'.")
        else:
            st.warning("Feature importance plot could not be generated.")

# --- Deepfake Detection Section ---
st.header("Deepfake Detection")
st.markdown("Upload an image to predict if it's a deepfake or real. Note: Compressed images (e.g., from WhatsApp) may lead to misclassifications due to quality loss. For best results, use high-quality images.")

# Image upload
uploaded_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
deepfake_threshold = 0.3  # Adjusted threshold to catch more deepfakes

# Function to preprocess image for deepfake detection
def preprocess_image(image):
    try:
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        # Resize to (299, 299)
        image = image.resize((299, 299))
        # Convert to array and preprocess
        image_array = np.array(image)
        image_array = preprocess_input(image_array)
        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)
        return image_array
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        st.stop()
        return None

# Deepfake prediction
if st.button("Predict Deepfake"):
    if uploaded_image is None:
        st.warning("Please upload an image to make a deepfake prediction.")
    else:
        # Load and preprocess image
        try:
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            image_array = preprocess_image(image)
            if image_array is None:
                st.stop()
            
            # Make prediction
            prob = deepfake_model.predict(image_array, verbose=0)[0][0]
            pred = 1 if prob > deepfake_threshold else 0
            label = "Deepfake" if pred == 1 else "Real"
            confidence = prob if label == "Deepfake" else 1 - prob
            
            # Display prediction
            st.subheader("Deepfake Prediction Result")
            st.write(f"**Prediction**: {label}")
            st.write(f"**Confidence**: {confidence:.4f}")
            st.write(f"**Raw Probability (Deepfake)**: {prob:.4f} (Threshold: {deepfake_threshold})")
        except Exception as e:
            st.error(f"Error making deepfake prediction: {e}")
            st.stop()

# Display EDA visualizations
st.header("Exploratory Data Analysis")
st.markdown("Below are key insights from the dataset used to train the fake news model. If images are missing, ensure they were generated by the training script and placed in the working directory. Copy them from the notebook's output directory if needed.")

# Check if EDA images exist and display them
eda_images = [
    'text_length_boxplot.png',
    'target_distribution.png',
    'fake_news_wordcloud.png',
    'real_news_wordcloud.png',
    'subject_distribution.png',
    'text_length_by_subject.png',
    'point_biserial_coefficient.png'
]

for img in eda_images:
    if os.path.exists(img):
        st.image(img, caption=img.replace('_', ' ').replace('.png', '').title(), use_column_width=True)
    else:
        st.warning(f"Image {img} not found. Please ensure it is in the working directory.")

# Display model performance
st.header("Model Performance")
st.markdown("""
**Fake News Detection (Logistic Regression)**:
- **Test F1 Score**: 0.6267
- **Test ROC AUC**: 0.7273
- The model uses a threshold of 0.4 to balance precision and recall.

**Deepfake Detection (Xception-based CNN)**:
- **Test F1 Score**: 0.8646
- **Test ROC AUC**: 0.7988
- The model uses a threshold of 0.3 to improve detection of deepfakes.
""")

# Instructions for use
st.header("How to Use")
st.markdown("""
1. **Fake News Detection**:
   - Enter a news article's title and text in the input fields.
   - Click "Predict Fake News" to see if the article is classified as fake or real.
   - Review the confidence score and feature importance plot.
2. **Deepfake Detection**:
   - Upload an image (JPG, JPEG, or PNG).
   - Click "Predict Deepfake" to see if the image is classified as deepfake or real.
   - Review the confidence score and raw probability.
3. Explore the EDA section to understand the fake news dataset characteristics.
""")