import streamlit as st

from utils import RAGChatbot

# Setting the page config
st.set_page_config(
    page_title="RAG Chatbot 🤖",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="auto",
)

# Adding title and description
st.title("RAG Chatbot 🤖")
st.markdown("""
Welcome to the **RAG Chatbot**! Upload your PDF file and ask questions about its content. 
The chatbot will provide answers based on the PDF document. 🚀
""")

# File uploader widget
uploaded_file = st.file_uploader("Choose a PDF file 📄", type="pdf")

# Processing the uploaded file
if uploaded_file is not None:
    local_path = f"/tmp/{uploaded_file.name}"
    with open(local_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"File {uploaded_file.name} uploaded successfully! 🎉")

    with st.spinner('Processing the PDF content... ⏳'):
        # Initializing and setting up the chatbot
        chatbot = RAGChatbot(local_path)
        chatbot.load_data()
        chatbot.pprint_first_page_content()
        chatbot.split_and_chunk()
        chatbot.add_to_vector_db()
        chatbot.initialize_llm()
        chatbot.create_retriever()
        chatbot.create_chain()

    st.success("PDF content has been processed and vector database has been created. 📚")

    # User input for chatbot interaction
    user_input = st.text_input("Ask a question to the chatbot: 💬")

    if user_input:
        with st.spinner('Getting the answer from the chatbot... ⏳'):
            response = chatbot.invoke_chain(user_input)
        st.write("Chatbot response:")
        st.write(response)

# Sidebar content
st.sidebar.title("About")
st.sidebar.info("""
This application uses the **RAG Chatbot** to process PDF documents and answer questions based on their content. 
Developed with ❤️ using Streamlit.
""")

st.sidebar.title("Contact")
st.sidebar.info("""
If you have any questions or feedback, please contact us at [emtinsajadi79@gmail.com].
""")