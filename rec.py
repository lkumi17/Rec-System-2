import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler, normalize
#import jpype
#jvm_path = "\Program Files\Java\jdk-22"
#jpype.startJVM(jvm_path)

def load_data():
    data = pd.read_csv('Dataset with image.csv')
    return data

data = load_data()

# Initialize tokenizer
from konlpy.tag import Okt
okt = Okt()

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

# Function to make recommendations
def recommend_report_design(input_criteria_1, input_criteria_2):
    combined_input = f"{input_criteria_1} {input_criteria_2}"
    input_text_vectorized = vectorizer.transform([preprocess_text(combined_input)])
    input_text_normalized = normalize(input_text_vectorized)
    text_similarities = cosine_similarity(input_text_vectorized, X)

    # Placeholder for the actual image feature vector of the input
    input_image_features = np.random.rand(1, image_features.shape[1])  # Replace with actual input
    input_image_normalized = normalize(input_image_features)
    image_similarities = cosine_similarity(input_image_features, image_features)

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
        preview_link = top_designs.iloc[i]['Image path']  # Assuming this is the column name for image links
        if i == 0:
            primary_recommendation = f"Recommended Report Design: {design_code}\n" \
                                     f"Similarity Score: {score:.2f}\n" \
                                     f"Preview Image: [Link to the {preview_link}]\n\n" \
                                     "Additional Suggestions:\n"
        else:
            recommendations.append(f"{i}. {design_code} (Score: {score:.2f}) - [Preview Link {preview_link}]")

    # Combine primary recommendation with additional suggestions
    return primary_recommendation + "\n".join(recommendations)

# Streamlit UI
st.title('Construction Design for Safety (DfS) Recommendation System')

input_criteria_1 = st.text_input("Enter the first criteria for the report:", "")
input_criteria_2 = st.text_input("Enter the second criteria for the report:", "")

if st.button('Recommend Design'):
    if input_criteria_1 and input_criteria_2:
        results = recommend_report_design(input_criteria_1, input_criteria_2)
        st.markdown(results, unsafe_allow_html=True)
    else:
        st.error('Please enter criteria to get a recommendation.')
