import os
import google.generativeai as genai
from dotenv import load_dotenv
from relevancy_processor_tfidf import preprocess_text, get_relevant_passages
from credentials_loader import CredentialsLoader

class Chatbot:
    def __init__(self, model_name, reference_file_path):
        load_dotenv()
        self.creds = CredentialsLoader.load_iam_creds()
        genai.configure(credentials=self.creds)
        self.model = genai.GenerativeModel(model_name=f'models/{model_name}')
        self.reference_file_path = reference_file_path

    def list_models(self):
        print('Available base models:', [m.name for m in genai.list_models()])
        print('Available tuned models:')
        try:
            tuned_models = genai.list_tuned_models()
            for i, model in enumerate(tuned_models):
                print(model.name)
                if i >= 4:  # Limit to first 5 models
                    break
        except Exception as e:
            print(f"Error listing tuned models: {e}")

    def generate_response(self, user_input):
        # Load the reference file
        if os.path.exists(self.reference_file_path):
            with open(self.reference_file_path, 'r') as file:
                reference_material = file.read()
                print("loaded reference")
        else:
            print(f"Warning: Reference file '{self.reference_file_path}' not found.")
            reference_material = ""

        # Get relevant passages and their similarity scores
        relevant_passages, top_scores = get_relevant_passages(user_input, reference_material, useVerbatim=False)

        # Print passages and their similarity scores
        # for i, (passage, score) in enumerate(zip(relevant_passages, top_scores)):
        #     print(f"Passage {i+1}:")
        #     print(passage)
        #     print(f"Similarity Score: {score:.4f}\n")

        # Filter passages with similarity scores below 0.2
        filtered_passages_list = [(i+1, passage, score) for i, (passage, score) in enumerate(zip(relevant_passages, top_scores)) if score >= 0.2]

        # Create a string for filtered passages with their similarity scores
        filtered_passages = ""
        for i, passage, score in filtered_passages_list:
            filtered_passages += f"Reference snippet:\n"
            filtered_passages += f"{passage}\n"
            filtered_passages += f"Similarity Score to prompt: {score:.4f}\n\n"

        # Create the full prompt
        full_prompt = f"""
            You're a helpful teaching assistant for a technical course on CS142 Distribted Computing. Answer the student's 'question' based on the given 'context' only.
            Your answer should ALWAYS be concise but sufficient, and quick to understand with proper formatting and equations when applicable. Make sure your response is at minimum 3 sentences/bullet points.
            The 'context' includes relevant snippets of text from the course contents, each with a similarity score to the 'question'
            
            ***
            'question' : {user_input}
            ***
            
            $$$
            'context' : {"".join(filtered_passages)}
            $$$
            
            Remember, you are a teaching assistant. Do not add any facts not provided in the 'context'. Provide the 'answer' in HTML format with CSS styled headers, including images and equations from the 'context' as relevant. For all headers use h1.
            For equations, please use the specific LaTeX math mode delimiters for your response, as following,
            inline math mode : `\(` and `\)`
            display math mode: insert linebreak after opening `$$`, `\[` and before closing `$$`, `\]`
        """

        # More aggressive prompting
        # full_prompt = f"""
        # MY PROMPT:
        # {user_input}

        # Relevant Reference Material:
        # {"".join(filtered_passages)}

        # RULES:
        # First, If I ask or say anything not loosely related to algorithms, computer science, distributed computing, or practical applications of CS etc. reply with "I am an LLM trained to answer questions for CS142 only."
        # Sometimes I could be referencing something seemingly unrelated but if it is still relevant in the context of distributed computing, continue.

        # You are a computer science professor with experience in distributed computing and CS theory. Please always try and reference the text material provided to help answer the prompt question.
        # Your answer should ALWAYS be concise and comprehensive, quick to understand with proper formatting and equations when applicable. Make sure your response is at minimum 3 sentences/bullet points.
        # ONLY IF the question topic/answer absolutely isn't specified in the reference material, then draw upon your own knowledge but make it clear to me that you did. If you used info from the reference, put (test:ref) at the end.

        # FINALLY, make sure EVERYTHING, including bullet points, formatting, equations/typesetting, etc. are outputted properly in PURE AND WELL DEFINED HTML using PROPER TAGS. NEVER USE * CHARACTERS IN YOUR RESPONSE, and do NOT retype the prompt question in your response. Thanks so much.
        # """
        
        print("GPT PROMPT")
        print(full_prompt)

        # Generate response using the model
        result = self.model.generate_content(full_prompt)
        print("GPT RESPONSE")
        print(result)
        
        return result.text
