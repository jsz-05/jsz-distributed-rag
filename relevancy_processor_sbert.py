from sentence_transformers import SentenceTransformer, util
import numpy as np
import re
import os

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
    
    # Load pre-trained sentence transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Encode the question and passages
    question_embedding = model.encode(processed_question, convert_to_tensor=True)
    passage_embeddings = model.encode(passages, convert_to_tensor=True)
    
    # Calculate cosine similarity between question and passages
    cosine_similarities = util.pytorch_cos_sim(question_embedding, passage_embeddings).flatten()
    
    # Get indices of top similar passages
    top_passage_indices = cosine_similarities.argsort(descending=True)[:num_passages]
    
    # Get the top passages and their similarity scores
    top_passages = [passages[i] for i in top_passage_indices]
    top_scores = [cosine_similarities[i].item() for i in top_passage_indices]
    
    return top_passages, top_scores



reference_file_path = 'corpus/new_reference.txt'

if os.path.exists(reference_file_path):
    with open(reference_file_path, 'r') as file:
        reference_material = file.read()

# Testing
question = "How does message passing work in distributed systems?"
reference_text = reference_material
top_passages, top_scores = get_relevant_passages(question, reference_text)

for i, (passage, score) in enumerate(zip(top_passages, top_scores)):
    print(f"Passage {i+1}:")
    print(passage)
    print(f"Similarity Score: {score:.4f}\n")
