# [RAG and Prompt Engineering](https://jsz.pythonanywhere.com/)

This repository contains the code and documentation for implementing retrieval-augmented generation and prompt engineering for a chatbot to aid students in distributed computing concepts. The current LLM used is Google's Gemini 1.5 Flash API via Google Cloud Console, with RAG using TF-IDF.

The chatbot in it's current work in progress state can be accessed [here.](https://jsz.pythonanywhere.com/)

---

To run the application locally, first ensure all dependencies/libraries are installed and API keys loaded in `.env`.

Optionally, run `scrape_website_enhanced` if you wish to update the corpus with the course website's current changes.

Then run `gemini_chatbot.py` or `gpt_chatbot.py` and navigate to http://127.0.0.1:5000/.

