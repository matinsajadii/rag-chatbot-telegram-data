from pprint import pprint

from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import (OnlinePDFLoader, PyPDFLoader,
                                                  UnstructuredPDFLoader)
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter


class RAGChatbot:
    def __init__(self, local_path, local_model="llama3"):
        self.local_path = local_path
        self.local_model = local_model
        self.data = None
        self.chunks = None
        self.vector_db = None
        self.llm = None
        self.retriever = None
        self.chain = None

    def load_data(self):
        if self.local_path:
            loader = PyPDFLoader(file_path=self.local_path)
            self.data = loader.load_and_split()
        else:
            raise ValueError('Upload a PDF file')

    def pprint_first_page_content(self):
        if self.data:
            pprint(self.data[0].page_content)
        else:
            raise ValueError('Data not loaded')

    def split_and_chunk(self):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
        self.chunks = text_splitter.split_documents(self.data)

    def add_to_vector_db(self):
        self.vector_db = Chroma.from_documents(
            documents=self.chunks,
            embedding=OllamaEmbeddings(model="nomic-embed-text", show_progress=True),
            collection_name='local-rag'
        )

    def initialize_llm(self):
        self.llm = ChatOllama(model=self.local_model)

    def create_retriever(self):
        QUERY_PROMPT = PromptTemplate(
            input_variables=["question"],
            template="""You are an AI language model assistant. Your task is to generate five
            different versions of the given user question to retrieve relevant documents from
            a vector database. By generating multiple perspectives on the user question, your
            goal is to help the user overcome some of the limitations of the distance-based
            similarity search. Provide these alternative questions separated by newlines.
            Original question: {question}""",
        )

        self.retriever = MultiQueryRetriever.from_llm(
            self.vector_db.as_retriever(),
            self.llm,
            prompt=QUERY_PROMPT
        )

    def create_chain(self):
        template = """Answer the question based ONLY on the following context:
        {context}
        Question:{question}"""

        prompt = ChatPromptTemplate.from_template(template)

        self.chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

    def invoke_chain(self, input_text):
        if self.chain:
            return self.chain.invoke(input_text)
        else:
            raise ValueError('Chain not created')

# Usage
local_path = "/mnt/c/Users/Green.PC/Documents/ASAMA/project/rag-chatbot-telegram-data/data/WEF_The_Global_Cooperation_Barometer_2024.pdf"
chatbot = RAGChatbot(local_path)
chatbot.load_data()
chatbot.pprint_first_page_content()
chatbot.split_and_chunk()
chatbot.add_to_vector_db()
chatbot.initialize_llm()
chatbot.create_retriever()
chatbot.create_chain()
response = chatbot.invoke_chain("")
print(response)
