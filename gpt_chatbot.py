import os
import openai
from flask import Flask, render_template, request, jsonify
from relevancy_processor import get_relevant_passages

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Load OpenAI credentials
openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)

# Reference file path
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

    # Filter passages with similarity scores below 0.2
    filtered_passages = [passage for passage, score in zip(relevant_passages, top_scores) if score >= 0.2]

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
     Your answer should ALWAYS be concise, and quick to understand with proper formatting and equations when applicable. Make sure your response is at minimum 3 sentences/bullet points.
     If you used info from the reference, put (test:ref) at the end.
     ONLY IF the question topic/answer absolutely isn't specified in the reference material, then draw upon your own knowledge but make it clear to me that you did.

     FINALLY, make sure EVERYTHING, including bullet points, formatting, equations/typesetting, etc. are outputted properly in PURE AND WELL DEFINED HTML using PROPER TAGS. NEVER USE * CHARACTERS IN YOUR RESPONSE, and do NOT retype the prompt question in your response
    """

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful computer science professor with experience in distributed computing and CS theory."},
            {"role": "user", "content": full_prompt}
        ],
        temperature=0.5,
        max_tokens=1500
    )

    result_text = response['choices'][0]['message']['content']

    return jsonify({'response': result_text})

if __name__ == '__main__':
    app.run(debug=True)