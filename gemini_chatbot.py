# First, set up API and all authentications to be able to use tuned models

import os
import google.generativeai as genai
from load_creds import load_creds, load_iam_creds
from relevancy_processor import preprocess_text, get_relevant_passages

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Load credentials
creds = load_iam_creds()

# Configure generative AI with credentials
genai.configure(credentials=creds)
print()
print('Available base models:', [m.name for m in genai.list_models()])
print()
print('Available tuned models:')
try:
    tuned_models = genai.list_tuned_models()
    for i, model in enumerate(tuned_models):
        print(model.name)
        if i >= 4:  # Limit to first 5 models
            break
except Exception as e:
    print(f"Error listing tuned models: {e}")

# Flask app setup
from flask import Flask, render_template, request, jsonify
import google.generativeai as genai

app = Flask(__name__)

# Model configuration
model_name = 'gemini-1.5-flash-001'
model = genai.GenerativeModel(model_name=f'models/{model_name}')
reference_file_path = 'corpus/new_reference.txt'

@app.route('/')
def distributed_chatbot():
    return render_template('chatbot.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json['message']
    
    if os.path.exists(reference_file_path):
        with open(reference_file_path, 'r') as file:
            reference_material = file.read()
    else:
        print(f"Warning: Reference file '{reference_file_path}' not found.")
        reference_material = ""
    
    # Get relevant passages and their similarity scores
    relevant_passages, top_scores = get_relevant_passages(user_input, reference_material)

    # Print passages and their similarity scores
    for i, (passage, score) in enumerate(zip(relevant_passages, top_scores)):
        print(f"Passage {i+1}:")
        print(passage)
        print(f"Similarity Score: {score:.4f}\n")

    # Filter passages with similarity scores below 0.2
    filtered_passages = [passage for passage, score in zip(relevant_passages, top_scores) if score >= 0.2]
    print(filtered_passages)

    # Create the full prompt
    full_prompt = f"""
    MY PROMPT:
    {user_input}

     Relevant Reference Material: 
    {"".join(filtered_passages)}

     RULES: 
     First, If I ask or say anything not loosely related to computer science, IGNORE everything below and reply with "I am an LLM trained to answer questions for CS142 only."
     
     You are a computer science professor with experience in distributed computing and CS theory. 
     Please always try and reference the text material provided to help answer the prompt question.
     Your answer should ALWAYS be concise but sufficient, and quick to understand with proper formatting and equations when applicable. Make sure your response is at minimum 3 sentences/bullet points.
     If you used info from the reference, put (test:ref) at the end.
     ONLY IF the question topic/answer absolutely isn't specified in the reference material, then draw upon your own knowledge but make it clear to me that you did.

     FINALLY, make sure EVERYTHING, including bullet points, formatting, equations/typesetting, etc. are outputted properly in PURE AND WELL DEFINED HTML using PROPER TAGS. NEVER USE * CHARACTERS IN YOUR RESPONSE, and do NOT retype the prompt question in your response
    """
    
    # If I ask for clarification, please do a quick recap of your previous answer, and feel free to use your own knowledge as a supplement to give the best, most clear, and easiest to understand follow up. 
    # If the question topic/answer/clarification request isn't specified in the reference material, DRAW UPON YOUR OWN KNOWLEDGE, and add (test_metric:ext) at the end of your response. 
    # Otherwise if...
    print("GPT PROMPT")
    print(full_prompt)

    # Generate response using the model
    result = model.generate_content(full_prompt)
    print("GPT RESPONSE")
    print(result)
    
    return jsonify({'response': result.text})


if __name__ == '__main__':
    app.run(debug=True)