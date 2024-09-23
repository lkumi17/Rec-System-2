import streamlit as st
import pandas as pd
import numpy as np
import os
import cv2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler, normalize
from konlpy.tag import Okt
#import jpype
#jvm_path = "\Program Files\Java\jdk-22"
#jpype.startJVM(jvm_path)

def load_data():
    data = pd.read_csv('Dataset with image.csv')
    return data

data = load_data()

# Initialize tokenizer
okt = Okt()

# Preprocess text function
def preprocess_text(text):
    return " ".join(okt.morphs(text))

# Apply preprocessing to each text column
for col in ['위험요소(제목)', '물적 발생형태', '인적 발생형태', '용도별', '공종별']:
    data[col] = data[col].apply(preprocess_text)

# Combine all text columns into one to represent each record
data['combined'] = data.apply(lambda row: ' '.join(row[['위험요소(제목)', '물적 발생형태', '인적 발생형태', '용도별', '공종별']]), axis=1)

# Vectorize the combined text
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['combined'])

# Normalize text features to unit norm
X_normalized = normalize(X)

# Convert stringified image feature lists into numpy arrays
data['image features'] = data['image features'].apply(eval)
image_features = np.array(data['image features'].tolist())

# Standardize and normalize image features
scaler = StandardScaler()
image_features_scaled = scaler.fit_transform(image_features)
image_features_normalized = normalize(image_features_scaled)

# Function to extract features from a single image
def extract_features_from_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise Exception(f"Error loading: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_contours = len(contours)

    hist_b = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([image], [1], None, [256], [0, 256])
    hist_r = cv2.calcHist([image], [2], None, [256], [0, 256])

    hist_b_norm = cv2.normalize(hist_b, hist_b).flatten()
    hist_g_norm = cv2.normalize(hist_g, hist_g).flatten()
    hist_r_norm = cv2.normalize(hist_r, hist_r).flatten()

    color_hist_norm = np.hstack((hist_b_norm, hist_g_norm, hist_r_norm))
    image_features_vector = np.hstack((num_contours, color_hist_norm))

    return image_features_vector

# Function to make recommendations
def recommend_report_design(input_criteria_1, input_criteria_2, input_image_path):
    combined_input = f"{input_criteria_1} {input_criteria_2}"
    input_text_vectorized = vectorizer.transform([preprocess_text(combined_input)])
    input_text_normalized = normalize(input_text_vectorized)

    # Extract features from the input image
    input_image_features = extract_features_from_image(input_image_path)
    input_image_normalized = normalize([input_image_features])

    # Combine text and image features
    combined_input_features = np.hstack((input_text_normalized.toarray(), input_image_normalized))
    combined_database_features = np.hstack((X_normalized.toarray(), image_features_normalized))

    # Calculate cosine similarity between combined features
    similarity_scores = cosine_similarity(combined_input_features, combined_database_features)
    top_indices = similarity_scores[0].argsort()[-5:][::-1]

    # Gather the top 5 recommendations
    top_designs = data.iloc[top_indices]
    top_scores = similarity_scores[0][top_indices]

    # Format the output
    recommendations = []
    for i, (index, score) in enumerate(zip(top_indices, top_scores)):
        design_code = top_designs.iloc[i]['Design code']
        preview_link = top_designs.iloc[i]['Image path']
        if i == 0:
            primary_recommendation = f"Recommended Report Design: {design_code}\n" \
                                     f"Similarity Score: {score:.2f}\n" \
                                     f"Preview Image: [Link to the {preview_link}]\n\n" \
                                     "Additional Suggestions:\n"
        else:
            recommendations.append(f"{i}. {design_code} (Score: {score:.2f}) - [Preview Link {preview_link}]")

    return primary_recommendation + "\n".join(recommendations)

# Streamlit UI
st.title('Construction Design for Safety (DfS) Recommendation System')

input_criteria_1 = st.text_input("Enter the first criteria for the report:", "")
input_criteria_2 = st.text_input("Enter the second criteria for the report:", "")
input_image = st.file_uploader("Upload an image for the design:", type=["jpg", "png", "jpeg"])

# Directory to save uploaded images temporarily
image_directory = '/tmp/recommendation_images/'
if not os.path.exists(image_directory):
    os.makedirs(image_directory)

if st.button('Recommend Design'):
    if input_criteria_1 and input_criteria_2 and input_image:
        input_image_path = os.path.join(image_directory, input_image.name)
        with open(input_image_path, 'wb') as f:
            f.write(input_image.getbuffer())

        try:
            results = recommend_report_design(input_criteria_1, input_criteria_2, input_image_path)
            st.markdown(results, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.error('Please enter criteria and upload an image to get a recommendation.')
