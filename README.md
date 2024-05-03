# AskPDF - LLM-driven PDF Question-Answering System

An LLM-based web application that takes multiple pdfs and can answer questions related to those pdfs. It uses Gemini so it requires a google API. <br /> 
Get your Google API here: https://aistudio.google.com/app/apikey <br />
Deployed on Streamlit. Here's the link for the web application: <br />
https://askpdf-llm-driven-pdf-question-answering-system.streamlit.app/  <br /><br />
![image](https://github.com/Animesh452/AskPDF-LLM-driven-PDF-Question-Answering-System/assets/68946005/4c85c8b7-aef6-4840-853c-e57b57708a39) <br />

To use this in VS code, follow these steps:
* Clone the repository.
* Create a virtual environment using `conda create -p venv python==3.10`. Gemini works well with Python version >= 3.10.
* Next `conda activate /venv`.
* Add a file named `.env`.
* Inside this write `GOOGLE_API_KEY="YOUR_API_KEY"`
* Type `pip install -r requirements.txt` in the terminal.
* Run the application using `streamlit run app.py`.
