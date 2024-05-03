import streamlit as st  # Import Streamlit for building the web app
from PyPDF2 import PdfReader  # Import PdfReader from PyPDF2 to read PDF files
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Import text splitter from LangChain
import os  # Import os module for interacting with the operating system
from langchain_google_genai import GoogleGenerativeAIEmbeddings  # Import Google Embeddings from LangChain
import google.generativeai as genai  # Import Google Generative AI package
from langchain_community.vectorstores import FAISS  # Import FAISS vector store from LangChain community package
from langchain_google_genai import ChatGoogleGenerativeAI  # Import Google Generative AI chat model from LangChain
from langchain.chains.question_answering import load_qa_chain  # Import question answering chain from LangChain
from langchain.prompts import PromptTemplate  # Import PromptTemplate from LangChain
from dotenv import load_dotenv  # Import load_dotenv from dotenv package to load environment variables

load_dotenv()  # Load environment variables from the .env file
os.getenv("GOOGLE_API_KEY")  # Get the Google API key from the environment variables
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))  # Configure the Google Generative AI package with the API key

# Function to get text from PDF files
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)  # Create a PdfReader object for each PDF file
        for page in pdf_reader.pages:
            text += page.extract_text()  # Extract text from each page and append to the text variable
    return text

# Function to split the text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)  # Create a text splitter
    chunks = text_splitter.split_text(text)  # Split the text into chunks
    return chunks

# Function to create a vector store from the text chunks
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")  # Create Google Embeddings
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)  # Create a FAISS vector store from the text chunks and embeddings
    vector_store.save_local("faiss_index")  # Save the vector store locally

# Function to create a conversational chain
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)  # Create a Google Generative AI chat model
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])  # Create a prompt template
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)  # Load a question answering chain
    return chain

# Function to handle user input and provide a response
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")  # Create Google Embeddings
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)  # Load the vector store
    docs = new_db.similarity_search(user_question)  # Perform similarity search on the user question
    chain = get_conversational_chain()  # Get the conversational chain
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)  # Get the response
    print(response)
    st.write("Reply: ", response["output_text"])  # Display the response on the Streamlit app

# Main function
def main():
    st.set_page_config("Chat PDF")  # Set the page config for the Streamlit app
    st.header("AskPDF - LLM-driven PDF Question Answering System💁")  # Display the header for the Streamlit app
    user_question = st.text_input("Ask a Question from the PDF Files")  # Get user input
    if user_question:
        user_input(user_question)  # Call the user_input function with the user question

    with st.sidebar:  # Create a sidebar
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)  # Allow users to upload multiple PDF files

        if st.button("Submit & Process"):  # Create a button to submit and process the PDF files
            with st.spinner("Processing..."):  # Display a spinner while processing
                raw_text = get_pdf_text(pdf_docs)  # Get the text from the PDF files
                text_chunks = get_text_chunks(raw_text)  # Split the text into chunks
                get_vector_store(text_chunks)  # Create a vector store from the text chunks
            st.success("Done")  # Display a success message

if __name__ == "__main__":
    main()