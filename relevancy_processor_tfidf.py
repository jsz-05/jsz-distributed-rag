from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

def preprocess_text(text):
    # Convert to lowercase and remove punctuation
    text = re.sub(r'[^\w\s]', '', text.lower())
    # Additional preprocessing
    return text

def split_into_sentences(text):
    # Split text into sentences
    sentences = re.split(r'(?<=[.!?]) +', text)
    return sentences

def create_passages_from_sentences(sentences, passage_sentences, overlap):
    passages = []
    for i in range(0, len(sentences) - passage_sentences + 1, passage_sentences - overlap):
        passage = ' '.join(sentences[i:i + passage_sentences])
        passages.append(passage)
    return passages

def get_relevant_passages(question, reference_text, num_passages=3, passage_sentences=20, overlap=2, useVerbatim=True):
    # Preprocess the question and reference text if useVerbatim is False
    if useVerbatim:
        processed_question = question
        processed_reference = reference_text
    else:
        processed_question = preprocess_text(question)
        processed_reference = preprocess_text(reference_text)
    
    # Split reference text into pages
    pages = re.split(r'\$\$\$\$\$ Content from .+ \$\$\$\$\$', reference_text)

    all_passages = []
    for page in pages:
        # Split page into sentences
        sentences = split_into_sentences(page)
        # Create overlapping passages from sentences
        passages = create_passages_from_sentences(sentences, passage_sentences, overlap)
        all_passages.extend(passages)

    # Check if there are passages to process
    if not all_passages:
        print("No passages found.")
        return [], []
    
    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_passages + [processed_question])
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")

    # Calculate cosine similarity between question and passages
    cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
    
    # Get indices of top similar passages
    top_passage_indices = cosine_similarities.argsort()[-num_passages:][::-1]
    
    # Get the top passages and their similarity scores
    top_passages = [all_passages[i] for i in top_passage_indices]
    top_scores = [cosine_similarities[i] for i in top_passage_indices]
    
    return top_passages, top_scores


import os

if __name__ == "__main__":
    reference_file_path = 'corpus/new_reference.txt'

    if os.path.exists(reference_file_path):
        with open(reference_file_path, 'r') as file:
            reference_material = file.read()

    # Testing
    question = "In a distributed system, can an agent refuse to receive a message?"
    question2 = "What is a queue in distributed computing"
    reference_text = reference_material
    top_passages, top_scores = get_relevant_passages(question, reference_text, useVerbatim=True)

    for i, (passage, score) in enumerate(zip(top_passages, top_scores)):
        print(f"Passage {i+1}:")
        print(passage)
        print(f"Similarity Score: {score:.4f}\n")
