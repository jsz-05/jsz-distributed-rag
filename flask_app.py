from flask import Flask, render_template, request, jsonify
from chatbot import Chatbot

app = Flask(__name__)
chatbot = Chatbot(model_name='gemini-1.5-flash', reference_file_path='corpus/new_reference.txt')

@app.route('/')
def distributed_chatbot():
    return render_template('chatbot.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json['message']
    response = chatbot.generate_response(user_input)
    return jsonify({'response': response})

if __name__ == '__main__':
    chatbot.list_models()
    app.run(debug=True)
