from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import os 

def preprocess_text(text):
    # Convert to lowercase and remove punctuation
    text = re.sub(r'[^\w\s]', '', text.lower())
    # Additional preprocessing
    return text

def get_relevant_passages(question, reference_text, num_passages=5, passage_length=750, overlap=0, similarity_threshold=0.7, useVerbatim=False):
    # Preprocess the question and reference text
    processed_question = preprocess_text(question)
    processed_reference = preprocess_text(reference_text)
    
    # Create overlapping passages
    passages = [reference_text[i:i+passage_length] for i in range(0, len(reference_text) - passage_length + 1, passage_length - overlap)]
    processed_passages = [processed_reference[i:i+passage_length] for i in range(0, len(processed_reference) - passage_length + 1, passage_length - overlap)]
    
    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(processed_passages + [processed_question])
    
    # Calculate cosine similarity between question and passages
    cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
    
    # Get indices of top similar passages
    top_passage_indices = cosine_similarities.argsort()[-num_passages*10:][::-1]
    
    # Deduplicate passages
    unique_passages = []
    unique_scores = []
    for idx in top_passage_indices:
        passage = passages[idx] if useVerbatim else processed_passages[idx]
        score = cosine_similarities[idx]
        if all(cosine_similarity(vectorizer.transform([preprocess_text(passage)]), vectorizer.transform([preprocess_text(p)]))[0][0] < similarity_threshold for p in unique_passages):
            unique_passages.append(passage)
            unique_scores.append(score)
        if len(unique_passages) >= num_passages:
            break
    
    return unique_passages, unique_scores



if __name__ == "__main__":
    reference_file_path = 'corpus/new_reference.txt'

    if os.path.exists(reference_file_path):
        with open(reference_file_path, 'r') as file:
            reference_material = file.read()

    # Testing
    question = "In a distributed system, can an agent refuse to receive a message?"
    question2 = "What is a queue in distributed computing"
    reference_text = reference_material
    top_passages, top_scores = get_relevant_passages(question2, reference_text, useVerbatim=True)

    for i, (passage, score) in enumerate(zip(top_passages, top_scores)):
        print(f"Passage {i+1}:")
        print(passage)
        print(f"Similarity Score: {score:.4f}\n")
