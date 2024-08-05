from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

def preprocess_text(text):
    # Convert to lowercase and remove punctuation
    text = re.sub(r'[^\w\s]', '', text.lower())
    # Additional preprocessing
    return text

def get_relevant_passages(question, reference_text, num_passages=5, passage_length=750, overlap=100):
    # Preprocess the question and reference text
    processed_question = preprocess_text(question)
    processed_reference = preprocess_text(reference_text)
    
    # Create overlapping passages
    passages = [processed_reference[i:i+passage_length] for i in range(0, len(processed_reference) - passage_length + 1, passage_length - overlap)]
    
    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(passages + [processed_question])
    
    # Calculate cosine similarity between question and passages
    cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
    
    # Get indices of top similar passages
    top_passage_indices = cosine_similarities.argsort()[-num_passages:][::-1]
    
    # Get the top passages and their similarity scores
    top_passages = [passages[i] for i in top_passage_indices]
    top_scores = [cosine_similarities[i] for i in top_passage_indices]
    
    return top_passages, top_scores

