# RAG Chatbot 🤖

Welcome to the **RAG Chatbot**! This application allows you to upload a PDF file and ask questions about its content. The chatbot will provide answers based on the PDF document using the Retrieval-Augmented Generation (RAG) approach.

<image src="https://images.ctfassets.net/ukazlt65o6hl/1rWCLaP3w1iMUSkZsg9dG4/ab39b7646762b37b2296c07ae08182ff/MicrosoftTeams-image__55_.png?w=750&h=386&fl=progressive&q=70&fm=jpg">

## Features

- Upload PDF files and process their content
- Ask questions based on the uploaded PDF content
- Retrieve relevant information using an AI-powered chatbot
- Interactive and user-friendly interface with Streamlit

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/matinsajadii/rag-chatbot.git
   cd rag-chatbot

2. **Install the required dependencies:**

   ```bash
   pip install -r requirements.txt

Ensure you have the following libraries installed:

- `streamlit`
- `pandas`
- `utils` (a custom module containing the **RAGChatbot** class)

3. **Run the application:**

   ```bash
   streamlit run app.py

## Usage

1. **Upload a PDF file:**

- Click on the "Choose a PDF file 📄" button.
- Select a PDF file from your local machine to upload.

2. **Processing the PDF:**

- Once uploaded, the application will process the PDF file and create a vector database from its content.

3. **Ask questions:**

- Enter your question in the text input box provided.
- The chatbot will generate a response based on the content of the uploaded PDF.

## Example

After running the application, you will see the following interface:

1. **Upload Section:**

2. **Chat Section:**

## About

This application uses the **RAG Chatbot** to process PDF documents and answer questions based on their content. Developed with ❤️ using Streamlit.

## Contact
If you have any questions or feedback, please contact us at emtinsajadi79@gmail.com.